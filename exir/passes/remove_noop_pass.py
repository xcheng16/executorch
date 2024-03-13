# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Tuple

import torch
from executorch.exir.pass_base import ExportPass, PassResult
from torch.fx import GraphModule

_DEQUANT_OPS: Tuple[torch._ops.OpOverload] = (
    torch.ops.quantized_decomposed.dequantize_per_tensor.default,
    torch.ops.quantized_decomposed.dequantize_per_channel.default,
)
_QUANT_OPS: Tuple[torch._ops.OpOverload] = (
    torch.ops.quantized_decomposed.quantize_per_tensor.default,
    torch.ops.quantized_decomposed.quantize_per_channel.default,
)


class RemoveNoopPass(ExportPass):
    """
    Removes noops that pass through arguments.
    """

    def remove_quantized_op(
        self, graph_module: GraphModule, node: torch.fx.Node
    ) -> None:
        node_input = list(node.args)[0]

        if not isinstance(node_input, torch.fx.Node):
            return

        # Let's assume that when entering this section of code the graph pattern is as follows:
        # Node A -> DQ -> slice_copy -> Q -> Node B. If the qparams of the DQ and Q are the same,
        # then after this the graph will look like this:
        # Node A -> Node B.
        if node_input.target in _DEQUANT_OPS:
            for user in list(node.users):
                if user.target in _QUANT_OPS:
                    # Drop the input arg and check that the qparams are the same.
                    qparams_dq = list(node_input.args)[1:]
                    qparams_q = list(user.args)[1:]
                    if qparams_dq != qparams_q:
                        return
                    user.replace_all_uses_with(node_input.args[0])

    def call(self, graph_module: GraphModule) -> PassResult:
        for node in graph_module.graph.nodes:
            if node.op != "call_function":
                continue

            if node.target not in (
                torch.ops.aten.to.dtype,
                torch.ops.aten.dropout.default,
                torch.ops.aten.slice_copy.Tensor,
            ):
                continue

            orig_tensor = node.args[0].meta["val"]

            if orig_tensor is node.meta["val"]:
                # If the graph is quantized, we must remove the entire pattern consisting of dq->op->q.
                # Otherwise, removing only the op will suffice.
                if node.args[0].target in _DEQUANT_OPS:
                    self.remove_quantized_op(graph_module, node)
                else:
                    node.replace_all_uses_with(node.args[0])
                continue

            if node.target == torch.ops.aten.slice_copy.Tensor:
                if orig_tensor.size() == node.meta["val"].size():
                    # If the graph is quantized, we must remove the entire pattern consisting of dq->op->q.
                    # Otherwise, removing only the op will suffice.
                    if node.args[0].target in _DEQUANT_OPS:
                        self.remove_quantized_op(graph_module, node)
                    else:
                        node.replace_all_uses_with(node.args[0])

        graph_module.graph.lint()
        graph_module.graph.eliminate_dead_code()
        return PassResult(graph_module, True)
