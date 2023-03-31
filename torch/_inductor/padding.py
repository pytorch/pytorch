import torch


def needs_padding(args):
    for arg in args:
        size = arg.meta["val"].shape
        for i in size:
            alignment_size = get_alignment_size(arg.meta["val"].dtype)
            if i % alignment_size != 0 or i < 512:
                return True
    return False


def larger_closest_multiple(n, k):
    if n % k == 0:
        return n
    else:
        return n + k - (n % k)


def get_alignment_size(dtype):
    # try:
    #     gpu_name = torch.cuda.get_device_name()
    # except RuntimeError:
    #     gpu_name = ""
    # if "A100" in gpu_name:
    #     has_a100 = True
    # else:
    #     has_a100 = False
    if dtype == torch.int8:
        return 16
    elif dtype in [torch.float16, torch.half, torch.bfloat16]:
        return 8
    elif dtype == torch.float32:
        return 4
    elif dtype == torch.float64:
        return 2
    else:
        return 1


def pad_and_slice_matrices(new_graph, a, b, pad_amount, batched=False):
    # Pad the first input matrix with zeroes
    new_a_pad = new_graph.call_function(
        torch.ops.aten.cat,
        (
            a,
            torch.ops.aten.zeros(
                (a.shape[0], a.shape[1], pad_amount)
                if batched
                else (a.shape[0], pad_amount)
            ),
            2 if batched else 1,
        ),
    )

    # Pad the second input matrix with zeroes
    new_b_pad = new_graph.call_function(
        torch.ops.aten.cat,
        (
            b,
            torch.ops.aten.zeros(
                (b.shape[0], pad_amount, b.shape[2])
                if batched
                else (pad_amount, b.shape[1])
            ),
            1 if batched else 0,
        ),
    )

    # Perform the matrix multiplication with the padded matrices
    new_mm_pad = new_graph.call_function(
        torch.ops.aten.bmm if batched else torch.ops.aten.matmul, (new_a_pad, new_b_pad)
    )

    # Slice the result to get the desired shape and not modify user code semantics
    new_mm = new_graph.call_function(
        torch.ops.aten.slice, (new_mm_pad, 2 if batched else 1, 0, a.shape[-1], 1)
    )

    return new_mm


def pad_mm(fx_g: torch.fx.GraphModule):
    # Leverages a classic interpreter pattern, thanks Horace!
    new_graph = torch.fx.Graph()
    env = {}
    for node in fx_g.graph.nodes:
        if node.target in [torch.ops.aten.addmm.default, torch.ops.aten.bmm]:
            # Currently this is a heuristic that decides if we should pad
            # Decided to only pad for medium size matrices and if alignment is off
            if needs_padding(node.args):
                size = int(tuple(env[node.args[0]].meta["tensor_meta"].shape)[0])
                alignment = get_alignment_size(
                    env[node.args[0]].meta["tensor_meta"].dtype
                )
                pad_amount = larger_closest_multiple(size, alignment) - size

                new_mm = pad_and_slice_matrices(
                    new_graph,
                    env[node.args[0]],
                    env[node.args[1]],
                    pad_amount,
                    batched=node.target == torch.ops.aten.bmm,
                )

            env[node] = new_mm

        else:
            new_node = new_graph.node_copy(node, lambda n: env[n])
            env[node] = new_node
    return torch.fx.GraphModule(fx_g, new_graph)
