# mypy: allow-untyped-defs
import functools
from collections import deque
from typing import Dict, List

import torch
from torch.utils._ordered_set import OrderedSet
from torch.utils._pytree import tree_map

from ..._dynamo.utils import counters
from ..ir import (
    ComputedBuffer,
    FixedLayout,
    FlexibleLayout,
    InputBuffer,
    StorageBox,
    Subgraph,
    TensorBox,
)
from ..lowering import lowerings
from ..pattern_matcher import (
    Arg,
    CallFunction,
    Match,
    PatternMatcherPass,
    register_graph_pattern,
)
from ..select_algorithm import (
    autotune_select_algorithm,
    ExternKernelChoice,
    TritonTemplate,
    TritonTemplateCaller,
)
from ..utils import ceildiv


B2B_GEMM_PASS = PatternMatcherPass(
    pass_name="b2b_gemm_pass",
)


def b2b_gemm_grid(M, P, meta):
    return (ceildiv(M, meta["BLOCK_SIZE_M"]) * ceildiv(P, meta["BLOCK_SIZE_P"]), 1, 1)


b2b_gemm_left_template = TritonTemplate(
    name="b2b_gemm_left",
    grid=b2b_gemm_grid,
    debug=False,
    source=r"""
{{def_kernel("A", "B", "C")}}


    # B2B_GEMM_LEFT_TRITON_ENTRANCE

    # dynamic shapes
    M = {{size("A", 0)}}
    N = {{size("A", 1)}}
    O = {{size("C", 0)}}
    P = {{size("C", 1)}}

    # dynamic strides
    stride_am = {{stride("A", 0)}}
    stride_an = {{stride("A", 1)}}
    stride_bn = {{stride("B", 0)}}
    stride_bo = {{stride("B", 1)}}
    stride_co = {{stride("C", 0)}}
    stride_cp = {{stride("C", 1)}}

    # output block counts
    num_m_block = tl.cdiv(M, BLOCK_SIZE_M)
    num_p_block = tl.cdiv(P, BLOCK_SIZE_P)

    # internal block counts
    num_n_block = tl.cdiv(N, BLOCK_SIZE_N)
    num_o_block = tl.cdiv(O, BLOCK_SIZE_O)

    # output block ids
    pid = tl.program_id(axis=0)
    m_block_id = pid // num_p_block
    p_block_id = pid % num_p_block

    # accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_P), dtype=tl.float32)

    # main loop
    offs_m = (m_block_id * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))
    offs_p = (p_block_id * BLOCK_SIZE_P + tl.arange(0, BLOCK_SIZE_P))
    # (subgraph(A @ B) @ C)
    offs_o = tl.arange(0, BLOCK_SIZE_O)
    for _ in range(num_o_block):
        c_mask = (offs_o[:, None] < O) & (offs_p[None, :] < P)
        c_ptrs = C + (offs_o[:, None] * stride_co + offs_p[None, :] * stride_cp)
        c = tl.load(c_ptrs, mask=c_mask, other=0.0).to(tl.float32)  # BLOCK_SIZE_O * BLOCK_SIZE_P
        acc_ab = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_O), dtype=tl.float32)
        offs_n = tl.arange(0, BLOCK_SIZE_N)
        for __ in range(num_n_block):
            a_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
            a_ptrs = A + (offs_m[:, None] * stride_am + offs_n[None, :] * stride_an)
            a = tl.load(a_ptrs, mask=a_mask, other=0.0).to(tl.float32)  # BLOCK_SIZE_M * BLOCK_SIZE_N
            b_mask = (offs_n[:, None] < N) & (offs_o[None, :] < O)
            b_ptrs = B + (offs_n[:, None] * stride_bn + offs_o[None, :] * stride_bo)
            b = tl.load(b_ptrs, mask=b_mask, other=0.0).to(tl.float32)  # BLOCK_SIZE_N * BLOCK_SIZE_O
            acc_ab += tl.dot(a, b, out_dtype=tl.float32)
            offs_n += BLOCK_SIZE_N
        # apply the subgraph
        {{ modification(
            subgraph_number=0,
            output_name="post_subgraph_acc_ab",
            inner_mm="acc_ab"
        ) | indent_except_first(2) }}
        acc += tl.dot(post_subgraph_acc_ab, c, out_dtype=tl.float32)
        offs_o += BLOCK_SIZE_O

    # type conversion
    acc = acc.to(tl.float16)

    # store preparation
    idx_m = offs_m[:, None]
    idx_p = offs_p[None, :]
    out_mask = (idx_m < M) & (idx_p < P)

    {{store_output(("idx_m", "idx_p"), "acc", "out_mask")}}
""",
)


b2b_gemm_right_template = TritonTemplate(
    name="b2b_gemm_right",
    grid=b2b_gemm_grid,
    debug=False,
    source=r"""
{{def_kernel("A", "B", "C")}}


    # B2B_GEMM_RIGHT_TRITON_ENTRANCE

    # dynamic shapes
    M = {{size("A", 0)}}
    N = {{size("A", 1)}}
    O = {{size("C", 0)}}
    P = {{size("C", 1)}}

    # dynamic strides
    stride_am = {{stride("A", 0)}}
    stride_an = {{stride("A", 1)}}
    stride_bn = {{stride("B", 0)}}
    stride_bo = {{stride("B", 1)}}
    stride_co = {{stride("C", 0)}}
    stride_cp = {{stride("C", 1)}}

    # output block counts
    num_m_block = tl.cdiv(M, BLOCK_SIZE_M)
    num_p_block = tl.cdiv(P, BLOCK_SIZE_P)

    # internal block counts
    num_n_block = tl.cdiv(N, BLOCK_SIZE_N)
    num_o_block = tl.cdiv(O, BLOCK_SIZE_O)

    # output block ids
    pid = tl.program_id(axis=0)
    m_block_id = pid // num_p_block
    p_block_id = pid % num_p_block

    # accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_P), dtype=tl.float32)

    # main loop (two cases)
    offs_m = (m_block_id * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))
    offs_p = (p_block_id * BLOCK_SIZE_P + tl.arange(0, BLOCK_SIZE_P))
    # (A @ subgraph(B @ C))
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    for _ in range(num_n_block):
        a_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        a_ptrs = A + (offs_m[:, None] * stride_am + offs_n[None, :] * stride_an)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0).to(tl.float32)  # BLOCK_SIZE_M * BLOCK_SIZE_N
        acc_bc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_P), dtype=tl.float32)
        offs_o = tl.arange(0, BLOCK_SIZE_O)
        for __ in range(num_o_block):
            b_mask = (offs_n[:, None] < N) & (offs_o[None, :] < O)
            b_ptrs = B + (offs_n[:, None] * stride_bn + offs_o[None, :] * stride_bo)
            b = tl.load(b_ptrs, mask=b_mask, other=0.0).to(tl.float32)  # BLOCK_SIZE_N * BLOCK_SIZE_O
            c_mask = (offs_o[:, None] < O) & (offs_p[None, :] < P)
            c_ptrs = C + (offs_o[:, None] * stride_co + offs_p[None, :] * stride_cp)
            c = tl.load(c_ptrs, mask=c_mask, other=0.0).to(tl.float32)  # BLOCK_SIZE_O * BLOCK_SIZE_P
            acc_bc += tl.dot(b, c, out_dtype=tl.float32)
            offs_o += BLOCK_SIZE_O
        # apply the subgraph
        {{ modification(
            subgraph_number=0,
            output_name="post_subgraph_acc_bc",
            inner_mm="acc_bc"
        ) | indent_except_first(2) }}
        acc += tl.dot(a, post_subgraph_acc_bc, out_dtype=tl.float32)
        offs_n += BLOCK_SIZE_N

    # type conversion
    acc = acc.to(tl.float16)

    # store preparation
    idx_m = offs_m[:, None]
    idx_p = offs_p[None, :]
    out_mask = (idx_m < M) & (idx_p < P)

    {{store_output(("idx_m", "idx_p"), "acc", "out_mask")}}
""",
)


# Note: load_ratio_left and load_ratio_right are only calculating numbers
# in the trivial subgraph case; i.e. (A @ (B @ C)) or ((A @ B) @ C)


def load_ratio_left(
    M: int, N: int, O: int, P: int, m: int, n: int, o: int, p: int
) -> float:
    """
    compute the ratio of estimated numbers of loads in baseline and b2bgemm
    M, N, O, P are matrix sizes
    m, n, o, p are block sizes
    |       | baseline (lower bound)        | b2bgemm
    | load  | M * N + N * O + M * O + O * P | M / m * P / p * O / o * (o * p + N / n * (m * n + n * o))
    | store | M * O + M * P                 | M * P
    b2bgemm is always better on stores, but for loads we need to find out beneficial cases using this function
    """
    base = M * N + N * O + M * O + O * P
    gemm = (
        ceildiv(M, m)
        * ceildiv(P, p)
        * ceildiv(O, o)
        * (o * p + ceildiv(N, n) * (m * n + n * o))
    )
    return base / gemm


def load_ratio_right(
    M: int, N: int, O: int, P: int, m: int, n: int, o: int, p: int
) -> float:
    """
    compute the ratio of estimated numbers of loads in baseline and b2bgemm
    M, N, O, P are matrix sizes
    m, n, o, p are block sizes
    |       | baseline (lower bound)        | b2bgemm
    | load  | N * O + O * P + M * N + N * P | M / m * P / p * N / n * (m * n + O / o * (n * o + o * p))
    | store | N * P + M * P                 | M * P
    b2bgemm is always better on stores, but for loads we need to find out beneficial cases using this function
    """
    base = N * O + O * P + M * N + N * P
    gemm = (
        ceildiv(M, m)
        * ceildiv(P, p)
        * ceildiv(N, n)
        * (m * n + ceildiv(O, o) * (n * o + o * p))
    )
    return base / gemm


# the block sizes are limited by hardware (the shared memory)
# intuitively, the optimization works when the intermediate matrix is large
# and we assign large block sizes to large dimensions
b2b_gemm_configs = [
    {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 16,
        "BLOCK_SIZE_O": 16,
        "BLOCK_SIZE_P": 16,
        "num_stages": 4,
        "num_warps": 8,
    },
    {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 32,
        "BLOCK_SIZE_O": 32,
        "BLOCK_SIZE_P": 32,
        "num_stages": 2,
        "num_warps": 4,
    },
    {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_O": 64,
        "BLOCK_SIZE_P": 64,
        "num_stages": 2,
        "num_warps": 4,
    },
    {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 16,
        "BLOCK_SIZE_O": 128,
        "BLOCK_SIZE_P": 16,
        "num_stages": 4,
        "num_warps": 8,
    },
    {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 32,
        "BLOCK_SIZE_O": 128,
        "BLOCK_SIZE_P": 32,
        "num_stages": 2,
        "num_warps": 4,
    },
    {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_O": 128,
        "BLOCK_SIZE_P": 64,
        "num_stages": 2,
        "num_warps": 4,
    },
    {
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 16,
        "BLOCK_SIZE_O": 16,
        "BLOCK_SIZE_P": 128,
        "num_stages": 4,
        "num_warps": 8,
    },
    {
        "BLOCK_SIZE_M": 32,
        "BLOCK_SIZE_N": 32,
        "BLOCK_SIZE_O": 32,
        "BLOCK_SIZE_P": 128,
        "num_stages": 2,
        "num_warps": 4,
    },
    {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_O": 64,
        "BLOCK_SIZE_P": 128,
        "num_stages": 2,
        "num_warps": 4,
    },
    {
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_O": 16,
        "BLOCK_SIZE_P": 128,
        "num_stages": 4,
        "num_warps": 8,
    },
    {
        "BLOCK_SIZE_M": 32,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_O": 32,
        "BLOCK_SIZE_P": 128,
        "num_stages": 2,
        "num_warps": 4,
    },
    {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_O": 64,
        "BLOCK_SIZE_P": 128,
        "num_stages": 2,
        "num_warps": 4,
    },
]


def is_b2b_gemm_good_on(
    is_left_assoc: bool,
    A_node: torch.fx.Node,
    B_node: torch.fx.Node,
    C_node: torch.fx.Node,
) -> bool:
    """
    checks whether the sizes are good for b2b_gemm
    """
    # basic checks
    if not all(["val" in A_node.meta, "val" in B_node.meta, "val" in C_node.meta]):
        return False
    fake_tensors = (
        A_node.meta["val"],
        B_node.meta["val"],
        C_node.meta["val"],
    )  # torch._subclasses.fake_tensor.FakeTensor

    A, B, C = fake_tensors

    def check_all_attr_true(objects, attr):
        return all(hasattr(obj, attr) and getattr(obj, attr) for obj in objects)

    if not check_all_attr_true(fake_tensors, "is_cuda") and not check_all_attr_true(
        fake_tensors, "is_xpu"
    ):
        return False
    if not all([len(A.shape) == 2, len(B.shape) == 2, len(C.shape) == 2]):
        return False
    if not ((A.shape[1] == B.shape[0]) and (B.shape[1] == C.shape[0])):
        return False
    # size checks: we only dispatch to B2B-GEMM when the average load ratio is > 1
    M, N = A.shape
    O, P = C.shape
    ratios = []
    if is_left_assoc:
        for config in b2b_gemm_configs:
            ratio = load_ratio_left(
                M,
                N,
                O,
                P,
                config["BLOCK_SIZE_M"],
                config["BLOCK_SIZE_N"],
                config["BLOCK_SIZE_O"],
                config["BLOCK_SIZE_P"],
            )
            ratios.append(ratio)
    else:
        for config in b2b_gemm_configs:
            ratio = load_ratio_right(
                M,
                N,
                O,
                P,
                config["BLOCK_SIZE_M"],
                config["BLOCK_SIZE_N"],
                config["BLOCK_SIZE_O"],
                config["BLOCK_SIZE_P"],
            )
            ratios.append(ratio)
    ratios.sort(reverse=True)
    average_ratio = 1.0
    for r in ratios[:3]:  # top 3 choices
        average_ratio *= r
    average_ratio = average_ratio ** (1 / 3)
    return (
        average_ratio > 1
    )  # even if average_ratio is close to 1, the number of stores is always better


def unoptimized_b2b_gemm(
    is_left_assoc: bool,
    subgraph: Subgraph,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    *,
    out: torch.Tensor,
) -> torch.Tensor:
    """
    The unoptimized version is used as a fallback when the b2b_gemm kernel is not beneficial.
    """
    if is_left_assoc:
        torch.mm(subgraph.graph_module(torch.mm(A, B)), C, out=out)
    else:
        torch.mm(A, subgraph.graph_module(torch.mm(B, C)), out=out)
    return out


unoptimized_choice = ExternKernelChoice(unoptimized_b2b_gemm)


def build_subgraph_buffer(
    args: List[TensorBox],
    subgraph: Subgraph,
):
    """
    This function is adapted from ../kernel/flex_attention.py.
    The goal is to take in the required args and produce the subgraph buffer
    The subgraph buffer is a ComputedBuffer that will be inlined into the triton template

    Args:
        args: The args that are passed into the subgraph
        subgraph: The Subgraph ir for which to produce the output node
    """
    cnt = 0
    env = {}
    for node in subgraph.graph_module.graph.nodes:
        if node.op == "placeholder":
            env[node] = args[cnt]
            cnt += 1
        elif node.op == "call_function":
            # For call_function we use the default lowerings and pass in the
            # already created TensorBoxes as args
            args, kwargs = tree_map(
                lambda x: env[x] if x in env else x, (node.args, node.kwargs)
            )
            env[node] = lowerings[node.target](*args, **kwargs)
        elif node.op == "output":

            def convert_output_node_to_buffer(output):
                if output is None:
                    return None
                output_node = output
                output_buffer = env[output_node]
                assert isinstance(output_buffer, TensorBox), (
                    "The output node for B2B-GEMM's subgraph must be a TensorBox, but got: ",
                    type(output_buffer),
                )
                assert isinstance(output_buffer.data, StorageBox), (
                    "The output node for B2B-GEMM's subgraph must be a StorageBox, but got: ",
                    type(output_buffer),
                )
                subgraph_buffer = ComputedBuffer(
                    name=None,
                    layout=FlexibleLayout(
                        device=output_buffer.data.get_device(),
                        dtype=output_buffer.data.get_dtype(),
                        size=output_buffer.data.get_size(),
                    ),
                    data=output_buffer.data.data,  # type: ignore[arg-type]
                )
                return subgraph_buffer

            # node.args[0] should be a single element representing the output of the subgraph
            return tree_map(convert_output_node_to_buffer, node.args[0])

    raise ValueError("B2B-GEMM was passed a subgraph with no output node!")


def create_placeholder(
    name: str, dtype: torch.dtype, device: torch.device
) -> TensorBox:
    """
    Creates a placeholder input buffers for producing subgraph_output
    """
    input_buffer = InputBuffer(name=name, layout=FixedLayout(device, dtype, [], []))
    return TensorBox.create(input_buffer)


def tuned_b2b_gemm(
    is_left_assoc: bool,
    subgraph: Subgraph,
    A: torch._inductor.ir.TensorBox,
    B: torch._inductor.ir.TensorBox,
    C: torch._inductor.ir.TensorBox,
    *,
    layout=None,
) -> torch._inductor.ir.TensorBox:
    # call .realize() to get rid of Pointwise
    A.realize()
    B.realize()
    C.realize()
    layout = FixedLayout(A.get_device_or_error(), A.get_dtype(), [A.shape[0], C.shape[1]])  # type: ignore[index]
    subgraph_buffer = build_subgraph_buffer(
        [create_placeholder("inner_mm", A.get_dtype(), A.get_device_or_error())],
        subgraph,
    )
    choices: list[TritonTemplateCaller] = []
    for config in b2b_gemm_configs:
        if is_left_assoc:
            b2b_gemm_left_template.maybe_append_choice(
                choices,
                input_nodes=(A, B, C),
                layout=layout,
                subgraphs=[subgraph_buffer],
                **config,
            )
        else:
            b2b_gemm_right_template.maybe_append_choice(
                choices,
                input_nodes=(A, B, C),
                layout=layout,
                subgraphs=[subgraph_buffer],
                **config,
            )
    # add the unoptimized choice to mitigate performance degradation
    choices.append(
        unoptimized_choice.bind(
            (A, B, C), layout, is_left_assoc=is_left_assoc, subgraph=subgraph
        )
    )
    # autotune
    return autotune_select_algorithm("b2b_gemm", choices, [A, B, C], layout)


# match the inner mm of a potential b2b_gemm
@register_graph_pattern(
    CallFunction(torch.ops.aten.mm, Arg(), Arg()),
    pass_dict=B2B_GEMM_PASS,
)
def b2b_gemm_handler(match: Match, mat1: torch.fx.Node, mat2: torch.fx.Node) -> None:
    # match.args: list[torch.fx.Node]

    def is_pointwise_node(node: torch.fx.Node) -> bool:
        return (
            node.op == "call_function"
            and isinstance(node.target, torch._ops.OpOverload)
            and (torch.Tag.pointwise in node.target.tags)
        )

    def is_mm(node: torch.fx.Node) -> bool:
        return node.target == torch.ops.aten.mm.default

    # the inner MM
    inner_mm = match.nodes[-1]

    # find the (candidate) outer MM, which will be re-checked below to ensure every path reaches it
    # In a real (A @ f(B @ C)), every path starting from (B @ C) must reach (A @ _).
    outer_mm = None
    node = inner_mm
    while len(node.users) > 0:
        node = next(iter(node.users))
        if is_mm(node):
            outer_mm = node
            break
        elif is_pointwise_node(node):
            continue
        else:
            break
    if not outer_mm:
        return

    # find the unique input node for outer_mm representing f(B @ C) in (A @ f(B @ C))
    # we call it the "f_node"
    # when the pattern is simply (A @ (B @ C)), f_node is just inner_mm
    f_node = inner_mm
    while next(iter(f_node.users)) is not outer_mm:
        f_node = next(iter(f_node.users))

    def all_reach_via_pointwise_with_no_other_inputs(
        src: torch.fx.Node,
        dst: torch.fx.Node,
    ) -> tuple[bool, OrderedSet[torch.fx.Node]]:
        """
        check whether every user path from src reaches dst via pointwise nodes,
        with no other input nodes for the intermediates and dst;
        return
        (1) the Boolean value
        (2) the subgraph node set including src and dst (which only makes sense when the Boolean value is True)
        """
        visited = OrderedSet[torch.fx.Node]()
        input_counter: Dict[torch.fx.Node, int] = {}

        all_reachable = True
        queue = deque([src])
        while queue:
            node = queue.popleft()
            if node not in visited:
                if node is dst:
                    visited.add(node)
                elif (node is src) or is_pointwise_node(node):
                    for user in node.users.keys():
                        # for nodes other than dst, bookkeep their users' input counts
                        if user not in input_counter:
                            input_counter[user] = len(user.all_input_nodes)
                        input_counter[user] -= 1
                        # continue BFS
                        queue.append(user)
                    visited.add(node)
                else:
                    all_reachable = False
                    break

        return (
            all_reachable and all(count == 0 for count in input_counter.values()),
            visited,
        )

    # check inner_mm reaches f_node on every user path via pointwise nodes with no outside input_nodes
    ok, subgraph_node_set = all_reach_via_pointwise_with_no_other_inputs(
        inner_mm, f_node
    )
    if not ok:
        return

    # check inner_mm's inputs and f_node's outputs
    if not (len(inner_mm.all_input_nodes) == 2 and len(f_node.users) == 1):
        return

    # at this point, the nodes between inner_mm and f_node (both included)
    # are all used internally inside (A @ subgraph(B @ C))
    # i.e. they neither have other users nor have other inputs

    # original graph and module
    graph, module = inner_mm.graph, inner_mm.graph.owning_module

    # construct the new (sub)graph
    subgraph_node_list: List[
        torch.fx.Node
    ] = []  # ordered list of nodes used for node removal later
    new_graph: torch.fx.Graph = torch.fx.Graph()
    node_remapping: Dict[torch.fx.Node, torch.fx.Node] = {}
    new_input_anchor: torch.fx.Node  # inner_mm, to be changed to an input node
    new_output_anchor: torch.fx.Node  # f_node, to be used to construct an output node
    new_input_node: torch.fx.Node
    new_output_node: torch.fx.Node
    for node in graph.nodes:  # preserve the order of nodes
        if node in subgraph_node_set:
            subgraph_node_list.append(node)
            new_node = new_graph.node_copy(
                node, lambda x: node_remapping[x] if x in node_remapping else x
            )
            node_remapping[node] = new_node
            if node is inner_mm:
                new_input_anchor = new_node
            if node is f_node:
                new_output_anchor = new_node
    if new_input_anchor is not new_output_anchor:  # subgraph is non-trivial
        # update the input node
        with new_graph.inserting_before(new_input_anchor):
            new_input_node = new_graph.placeholder(name="subgraph_input")
            new_input_node.meta.update(new_input_anchor.meta)
            new_input_anchor.replace_all_uses_with(new_input_node)
        new_graph.erase_node(new_input_anchor)
        # add the output node
        new_output_node = new_graph.output(new_output_anchor)
        new_output_node.meta.update(new_output_anchor.meta)
    else:  # subgraph is trivial, e.g. (A @ (B @ C))
        # update the input node
        with new_graph.inserting_before(new_input_anchor):
            new_input_node = new_graph.placeholder(name="subgraph_input")
            new_input_node.meta.update(new_input_anchor.meta)
            new_input_anchor.replace_all_uses_with(new_input_node)
        new_graph.erase_node(new_input_anchor)
        # update the output node (don't use new_output_anchor since it has been erased)
        new_output_node = new_graph.output(new_input_node)
        new_output_node.meta.update(new_input_node.meta)
    new_graph.lint()

    # construct the subgraph
    subgraph = Subgraph(
        name="subgraph", graph_module=torch.fx.GraphModule(module, new_graph)
    )

    # two cases
    # (1) (subgraph(A @ B) @ C), called "left_assoc"
    # (2) (A @ subgraph(B @ C)), called "right_assoc"
    is_left_assoc = outer_mm.args[0] is f_node

    # find the nodes A, B, C and check the sizes
    A: torch.fx.Node
    B: torch.fx.Node
    C: torch.fx.Node
    if is_left_assoc:
        A = inner_mm.args[0]  # type: ignore[assignment]
        B = inner_mm.args[1]  # type: ignore[assignment]
        C = outer_mm.args[1]  # type: ignore[assignment]
    else:
        A = outer_mm.args[0]  # type: ignore[assignment]
        B = inner_mm.args[0]  # type: ignore[assignment]
        C = inner_mm.args[1]  # type: ignore[assignment]
    if not is_b2b_gemm_good_on(is_left_assoc, A, B, C):
        return

    # finally update the original graph
    counters["inductor"]["b2b_gemm"] += 1
    graph = match.graph
    with graph.inserting_before(outer_mm):
        function = functools.partial(tuned_b2b_gemm, is_left_assoc, subgraph)
        function.__name__ = tuned_b2b_gemm.__name__  # type: ignore[attr-defined]
        function._inductor_lowering_function = True  # type: ignore[attr-defined]
        replacement: torch.fx.Node = graph.call_function(
            function,
            (A, B, C),
            match.kwargs,
        )
        replacement.meta.update(outer_mm.meta)
        outer_mm.replace_all_uses_with(replacement)
    # erase unnecessary nodes
    graph.erase_node(outer_mm)
    for node in reversed(subgraph_node_list):
        graph.erase_node(node)
    graph.lint()
