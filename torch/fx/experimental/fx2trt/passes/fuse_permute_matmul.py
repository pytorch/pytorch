import warnings
import torch
import torch.fx
import torch.fx.experimental.fx_acc.acc_ops as acc_ops


def trt_transposed_matmul(lhs: torch.Tensor, rhs: torch.Tensor, lhs_transposed: bool, rhs_transposed: bool):
    if lhs_transposed:
        lhs = lhs.transpose(-1, -2)
    if rhs_transposed:
        rhs = rhs.transpose(-1, -2)
    return torch.matmul(lhs, rhs)


def fuse_permute_matmul(gm: torch.fx.GraphModule):
    """
    Fuse pattern like permute + matmul if permute is transposing the last two dimension.
    """

    def check_permute(node: torch.fx.Node):
        ranks = len(node.meta["tensor_meta"].shape)
        permutation = list(i % ranks for i in node.kwargs["permutation"])  # type: ignore[union-attr]
        allowed_permutation = list(i for i in range(ranks))
        allowed_permutation[-1] = ranks - 2
        allowed_permutation[-2] = ranks - 1
        return len(node.users) == 1 and permutation == allowed_permutation

    for node in gm.graph.nodes:
        if node.target == acc_ops.matmul:
            lhs, rhs = node.kwargs["input"], node.kwargs["other"]
            lhs_transposed = rhs_tranposed = False

            if lhs.target == acc_ops.permute and check_permute(lhs):
                lhs_transposed = True
                lhs = lhs.kwargs["input"]

            if rhs.target == acc_ops.permute and check_permute(rhs):
                rhs_tranposed = True
                rhs = rhs.kwargs["input"]

            if lhs_transposed or rhs_tranposed:
                with gm.graph.inserting_before(node):
                    fused_node = gm.graph.call_function(trt_transposed_matmul, args=(lhs, rhs, lhs_transposed, rhs_tranposed))
                node.replace_all_uses_with(fused_node)

    gm.graph.eliminate_dead_code()
    gm.recompile()
    return gm


try:
    import tensorrt as trt
    from torch.fx.experimental.fx2trt.fx2trt import tensorrt_converter
except Exception:
    warnings.warn("Unable to import TensorRT related libraries.")
else:
    @tensorrt_converter(trt_transposed_matmul)
    def trt_transposed_matmul_converter(network, target, args, kwargs, name):
        lhs, rhs, lhs_transposed, rhs_transposed = args

        for i in [lhs, rhs]:
            if not isinstance(i, trt.tensorrt.ITensor):
                raise RuntimeError(
                    f"trt_transposed_matmul received input {i} that is not part "
                    "of the TensorRT region!"
                )

        layer = network.add_matrix_multiply(
            lhs,
            trt.MatrixOperation.TRANSPOSE if lhs_transposed else trt.MatrixOperation.NONE,
            rhs,
            trt.MatrixOperation.TRANSPOSE if rhs_transposed else trt.MatrixOperation.NONE,
        )
        layer.name = name
        return layer.get_output(0)
