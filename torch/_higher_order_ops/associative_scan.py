# mypy: allow-untyped-defs
import functools
import itertools
import operator
from collections.abc import Callable
from typing import Any

import torch
import torch._prims_common as utils
import torch.utils._pytree as pytree
from torch._C import DispatchKey
from torch._functorch.vmap import unwrap_batched, wrap_batched
from torch._higher_order_ops.utils import (
    _batch_dims_as_last_for_scan,
    _maybe_compile_and_run_fn,
    _maybe_run_with_interpreter,
    _move_batch_dims_to_last_for_scan,
    _VmapCombineFnWrapper,
    check_input_alias_and_mutation_return_outputs,
    check_meta_consistency,
    create_bw_fn,
    first_slice_copy,
    first_slice_copy_with_grad,
    materialize_as_graph,
    reenter_make_fx,
    register_fake,
    save_values_for_backward,
    saved_values,
    split_into_chunks,
    unique_graph_id,
    validate_subgraph_args_types,
)
from torch._ops import HigherOrderOperator
from torch.fx.experimental.proxy_tensor import (
    disable_proxy_modes_tracing,
    ProxyTorchDispatchMode,
    track_tensor_tree,
)
from torch.fx.experimental.symbolic_shapes import statically_known_true


aten = torch._ops.ops.aten


def wrap_combine_fn_flat(*args, combine_fn, spec, num_leaves):
    if len(args) != 2 * num_leaves:
        raise AssertionError(
            f"Combine_fn received wrong number of arguments, expected {2 * num_leaves}, but got {len(args)}"
        )
    lhs = pytree.tree_unflatten(args[:num_leaves], spec)
    rhs = pytree.tree_unflatten(args[num_leaves:], spec)
    return combine_fn(lhs, rhs)


def _interleave(a, b, dim=0):
    # https://stackoverflow.com/questions/60869537/how-can-i-interleave-5-pytorch-tensors
    if b_trunc := (a.shape[dim] == b.shape[dim] + 1):
        pad = (
            [0] * ((b.ndim - dim - 1) * 2 + 1)
            + [1]
            + [0] * (b.ndim * 2 - ((b.ndim - dim - 1) * 2 + 2))
        )
        b = torch.nn.functional.pad(b, pad)

    stacked = torch.stack([a, b], dim=dim + 1)
    interleaved = torch.flatten(stacked, start_dim=dim, end_dim=dim + 1)
    # pyrefly: ignore [unbound-name]
    if b_trunc:
        # TODO: find torch alternative for slice_along dim for torch.jit.script to work
        interleaved = aten.slice(interleaved, dim, 0, b.shape[dim] + a.shape[dim] - 1)
    return interleaved


def safe_map(f, *args):
    args = list(map(list, args))
    n = len(args[0])
    for arg in args[1:]:
        if len(arg) != n:
            raise ValueError("length mismatch: {list(map(len, args))}")

    def nf(a):
        return f(*a)

    return list(map(nf, zip(*args)))


class AssociativeScanOp(HigherOrderOperator):
    def __init__(self):
        super().__init__("associative_scan")

    def __call__(self, combine_fn, xs, additional_inputs):
        # There is currently an issue that the ScanOp is sometimes called with
        # the additional_inputs being a list. See https://github.com/pytorch/pytorch/issues/145785
        # Once this issue is resolved, the assertion should only allow tuples
        # and the tuple cast should be removed
        if not isinstance(additional_inputs, (tuple, list)):
            raise AssertionError(
                f"additional_inputs must be a tuple or list, got {type(additional_inputs)}"
            )
        additional_inputs = (
            tuple(additional_inputs)
            if isinstance(additional_inputs, list)
            else additional_inputs
        )
        validate_subgraph_args_types(additional_inputs)
        # pyrefly: ignore [missing-attribute]
        return super().__call__(combine_fn, xs, additional_inputs)

    # pyrefly: ignore [bad-override]
    def gen_schema(self, combine_fn, xs, additional_inputs):
        from torch._higher_order_ops.schema import HopSchemaGenerator

        # For associative scan, we need two copies of xs for the combine function
        # The combine function takes two elements and returns one element
        xs_slice1 = [first_slice_copy(x) for x in xs]
        xs_slice2 = [first_slice_copy(x) for x in xs]
        all_inputs = tuple(xs_slice1 + xs_slice2 + list(additional_inputs))

        combine_gm: torch.fx.GraphModule = materialize_as_graph(combine_fn, all_inputs)
        (
            _,
            _,
            _,
            mutated_inputs,
            outputs,
        ) = check_input_alias_and_mutation_return_outputs(combine_gm)
        if len(mutated_inputs) > 0:
            raise RuntimeError(
                "For associative_scan, combine_fn cannot have in-place mutations but found "
                f"{mutated_inputs}-th inputs are mutated."
            )

        schema_gen = HopSchemaGenerator(self)
        schema_gen.add_arg("combine_fn", combine_gm)

        for idx, x in enumerate(xs):
            schema_gen.add_arg(f"xs{idx}", x)

        for idx, arg in enumerate(additional_inputs):
            schema_gen.add_arg(
                f"additional_input{idx}",
                arg,
            )

        for out in outputs:
            schema_gen.add_output(out)

        schema_gen.add_schema_tree_spec(combine_fn, xs, additional_inputs)
        return schema_gen.gen_schema()


associative_scan_op = AssociativeScanOp()


def associative_scan(
    combine_fn: Callable[[pytree.PyTree, pytree.PyTree], pytree.PyTree],
    xs: pytree.PyTree,
    dim: int,
    reverse: bool = False,
    combine_mode: str = "pointwise",
) -> torch.Tensor:
    r"""
    Performs an inclusive scan with an associative combine function.

    .. warning::

        ``torch.associative_scan`` is a prototype feature in PyTorch. It currently
        does not support autograd and you may run into miscompiles.
        Read more about feature classification at:
        https://pytorch.org/blog/pytorch-feature-classification-changes/#prototype

    With ``combine_mode="pointwise"``, efficient execution requires runtime code
    generation via ``torch.compile``, and codegen is only supported on backends
    with scan support (currently CUDA and XPU). On other devices the operator
    still runs eagerly via the generic fallback.

    Args:
        combine_fn (Callable): A binary callable with type ``(Tensor, Tensor) -> Tensor``,
            or if input is a pytree ``(pytree, pytree) -> pytree``.
            This function must be pure, satisfy the associative property and have no
            side-effects. It may close over lifted arguments (e.g. freevars). On the
            autograd path in eager mode, tensor freevars are permitted as long as they do
            not require gradients (gradients for lifted arguments are not supported). Under
            ``torch.compile`` with ``backend="inductor"`` tensor freevars are still rejected
            outright; only ``int``/``SymInt`` lifted arguments are supported there.
        xs (torch.Tensor): The input tensor, or nested pytree of tensors.
        dim (int): the dimension to scan over
        reverse (bool): A boolean stating if the scan should be reversed with respect to ``dim``, default ``False``.
        combine_mode (str): A string indicating whether the ``combine_fn`` is ``pointwise`` or ``generic``, default ``pointwise``.
            If ``combine_mode=pointwise``, ``combine_fn`` must be pure and may only contain pointwise
            operations; under ``torch.compile`` ``xs`` must be on a backend with scan codegen support
            (CUDA or XPU), otherwise the generic fallback is used.
            In all other cases ``combine_mode=generic`` should be used.
            Note: ``combine_mode=pointwise`` is more efficient than ``combine_mode=generic``.

    Returns:
        A pytree of the same structure and shape as ``xs``. If the scan dimension has size 0,
        the output mirrors the (empty) input unchanged. The gradient with respect to ``xs``
        is also empty (size 0 along ``dim``), since there are no elements to differentiate through.

    Example::

        def add(x: torch.Tensor, y: torch.Tensor):
            return x + y


        cumsum = associative_scan(add, x, dim)

    """
    # TODO: Support lifted arguments in inductor for associative_scan
    # TODO: Support autograd for cases with lifted arguments for combine_mode=pointwise

    # The reason we flatten xs before calling into dynamo is that
    # we want to create a consistent input ordering for combine_fn
    # and we also want to the input ordering matches the output ordering.
    leaves_xs_orig, spec_xs = pytree.tree_flatten(xs)

    def _validate_input(cfn, lxs, d, r, cm):
        # Basic arguments check
        if not callable(cfn):
            raise ValueError(f"Combine_fn must be a callable, but got {cfn}")
        if not isinstance(d, int):
            raise ValueError("Dim must be an int, but got " + str(type(d)))
        if not isinstance(r, bool):
            raise RuntimeError("Reverse must be a bool, but got " + str(type(r)))
        if cm not in ["pointwise", "generic"]:
            raise ValueError(
                f"Combine_mode must either 'pointwise' or 'generic', but got {cm}"
            )

        # Checks for xs
        if len(lxs) == 0:
            raise ValueError("Expected at least 1 xs leaf")
        if any(not isinstance(x, torch.Tensor) for x in lxs):
            raise ValueError("xs leaves must be a Tensor")
        if any(x.is_sparse for x in lxs):
            raise ValueError(
                "xs leaves must dense Tensors, consider using `to_dense()`"
            )
        if any(x.ndim <= d for x in lxs):
            raise ValueError("All xs leaves must have at least 'dim + 1' dimensions")

    ndim = leaves_xs_orig[0].ndim
    dim = utils.canonicalize_dim(ndim, dim)

    _validate_input(combine_fn, leaves_xs_orig, dim, reverse, combine_mode)

    # Move scan dim to 0 and always perform scan on dim 0
    leaves_xs = [torch.movedim(elem, dim, 0) for elem in leaves_xs_orig]

    if reverse:
        leaves_xs = [torch.flip(elem, [0]) for elem in leaves_xs]

    if combine_mode == "generic":
        # The generic_associative_scan implementation calls the combine_fn with a `batch` along the scan dimension
        # For example, consider:
        # def add(x: torch.Tensor, y: torch.Tensor):
        #     return x + y
        # leaves = torch.tensor([[0.0, 1.0, 2.0, 3.0]
        #                        [0.0, 1.0, 2.0, 3.0]])
        # which has shape 2 x 4;
        # dim = 1;
        # In the first iteration of `_scan` the combine_fn gets invoked with
        # combine_fn([torch.tensor([[0.0, 2.0],
        #                           [0.0, 2.0]])],
        #            [torch.tensor([[1.0, 3.0],
        #                           [1.0, 3.0]])])
        # The arguments are of shape 2 x 2, but can be evaluated in parallel along the scan dimension.
        combine_fn = functools.partial(
            wrap_combine_fn_flat,
            combine_fn=torch.vmap(
                combine_fn,
                in_dims=(
                    pytree.tree_unflatten([0] * len(leaves_xs), spec_xs),
                    pytree.tree_unflatten([0] * len(leaves_xs), spec_xs),
                ),
                out_dims=0,
            ),
            spec=spec_xs,
            num_leaves=len(leaves_xs),
        )
        out = generic_associative_scan(combine_fn, leaves_xs, additional_inputs=())
        out = pytree.tree_unflatten(out, spec_xs)
    else:
        combine_fn = functools.partial(
            wrap_combine_fn_flat,
            combine_fn=combine_fn,
            spec=spec_xs,
            num_leaves=len(leaves_xs),
        )

        def run_flattened_associative_scan(combine_fn, leaves_xs):
            return associative_scan_op(combine_fn, leaves_xs, additional_inputs=())

        out = _maybe_compile_and_run_fn(
            run_flattened_associative_scan,
            combine_fn,
            leaves_xs,
        )

    if reverse:
        out = pytree.tree_map(lambda elem: elem.flip([0]), out)

    out = pytree.tree_map(lambda elem: torch.movedim(elem, 0, dim), out)

    return out


def generic_associative_scan(operator, leaves, dim=0, additional_inputs=()):
    r"""
    This function performs the associative_scan operation.
    The algorithm works by recursively collecting neighbours of ``leaves`` and subsequently
    applying the ``operator`` on all pairs in parallel along ``dim``.
    The results of the recursive calls are later combined.

    Args:
        operator (Callable): A binary callable with type ``(Tensor, Tensor) -> Tensor``,
            or if input is a pytree ``(pytree, pytree) -> pytree``.
            This function must be pure, pointwise, and satisfy the associative property.
        leaves (torch.Tensor): A list of torch.Tensors converted from the pytree of
            ``xs`` provided to ``associative_scan``.
            All inputs are expected to have the same shape.
        dim (int): the dimension to scan over
        additional_inputs (Tuple of tensors): A tuple of lifted parameters from the global scope.
            This parameter will be populated internally.

    Example::

        def add(x: torch.Tensor, y: torch.Tensor):
            return x + y

        leaves = torch.tensor([0.0, 1.0, 2.0, 3.0])

        First iteration of _scan ->
            # odd_elems -> apply operator on all neighbours
            # odd_elems = operator([torch.tensor([0.0, 2.0])],
            #                      [torch.tensor([1.0, 3.0])])
            odd_elems = torch.tensor([1.0, 5.0])
            Second iteration of _scan ->
                # odd_elems = operator([torch.tensor([1.0])],
                #                      [torch.tensor([5.0])])
                odd_elems = torch.tensor([6.0])
                # even_elems -> apply operator on all odd_elems and
                # every second element of ``elems``, starting from the second element.
                # even_elems is expanded with the first element of ``elems``
                even_elems = [1.0]
                # Merges odd_elems and even_elems
                res = torch.tensor([1.0, 6.0])
            # even_elems -> apply operator on all odd_elems and
            # every second element of ``elems``, starting from the second element.
            # even_elems is expanded with the first element of ``elems``
            even_elems = [0.0, 3.0]
            # Merges odd_elems and even_elems
            res = torch.tensor([0.0, 1.0, 3.0, 6.0])

    """

    def call_operator(*args):
        return pytree.tree_leaves(operator(*args))

    def _scan(elems):
        """Perform the actual recursive scan on ``elems``."""
        num_elems = elems[0].shape[dim]

        if num_elems < 2:
            return elems

        reduced_elems = call_operator(
            *[aten.slice(elem, dim, 0, -1, 2) for elem in elems],
            *[aten.slice(elem, dim, 1, None, 2) for elem in elems],
            *additional_inputs,
        )

        # Recursively compute scan for partially reduced tensors.
        odd_elems = _scan(reduced_elems)

        if num_elems % 2 == 0:
            even_elems = call_operator(
                *[aten.slice(e, dim, 0, -1) for e in odd_elems],
                *[aten.slice(e, dim, 2, None, 2) for e in elems],
                *additional_inputs,
            )
        else:
            even_elems = call_operator(
                *odd_elems,
                *[aten.slice(e, dim, 2, None, 2) for e in elems],
                *additional_inputs,
            )

        # The first element of a scan is the same as the first element
        # of the original `elems`.
        even_elems = [
            torch.cat([aten.slice(elem, dim, 0, 1), result], dim=dim)
            if result.shape.numel() > 0 and elem.shape[dim] > 0
            else result
            if result.shape.numel() > 0
            else aten.slice(
                elem, dim, 0, 1
            )  # Jax allows/ignores concat with 0-dim, Pytorch does not
            for (elem, result) in zip(elems, even_elems)
        ]

        return list(
            safe_map(functools.partial(_interleave, dim=dim), even_elems, odd_elems)
        )

    scans = _scan(leaves)

    return scans


def trace_associative_scan(
    proxy_mode,
    func_overload,
    combine_fn: Callable,
    xs: list[torch.Tensor],
    additional_inputs: tuple[torch.Tensor],
):
    from torch._dynamo.utils import clone_input

    with disable_proxy_modes_tracing():
        sample_xs = [first_slice_copy(x) for x in itertools.chain(xs, xs)]
        sample_additional_inputs = [
            clone_input(x) if isinstance(x, torch.Tensor) else x
            for x in additional_inputs
        ]
        combine_graph = reenter_make_fx(combine_fn)(
            *sample_xs, *sample_additional_inputs
        )

    outputs = None
    for node in combine_graph.graph.nodes:
        if node.op == "output":
            if outputs is not None:
                raise AssertionError("found multiple output nodes in combine_graph")
            if len(node.args) != 1:
                raise AssertionError(
                    f"expected output node to have 1 arg, got {len(node.args)}"
                )
            outputs = node.args[0]

    if outputs is None:
        raise AssertionError("no output node found in combine_graph")
    outputs = pytree.tree_leaves(outputs)
    if len(outputs) != len(xs):
        raise AssertionError(
            f"expected combine_fn to return {len(xs)} results but got {len(outputs)}"
        )

    xs_fake_tensors: list[torch.Tensor | torch.SymInt | int] = [
        first_slice_copy(x) for x in xs
    ]
    output_fake_tensors: list[torch.Tensor | torch.SymInt | int] = [
        c.meta["val"] for c in outputs
    ]
    check_meta_consistency(
        xs_fake_tensors, output_fake_tensors, "init", "carry", include_contiguity=False
    )

    _, combine_graph_name = unique_graph_id(
        proxy_mode, prefix="associative_scan_combine_graph"
    )

    proxy_mode.tracer.root.register_module(combine_graph_name, combine_graph)

    args = (combine_graph, xs, additional_inputs)
    proxy_args = pytree.tree_map(proxy_mode.tracer.unwrap_proxy, args)
    out_proxy = proxy_mode.tracer.create_proxy(
        "call_function", func_overload, proxy_args, {}, name="associative_scan"
    )

    with disable_proxy_modes_tracing():
        out = tuple(aten.clone(x) for x in xs)

    return track_tensor_tree(out, out_proxy, constant=None, tracer=proxy_mode.tracer)


@associative_scan_op.py_impl(DispatchKey.CompositeExplicitAutograd)
def associative_scan_op_dense(combine_fn, xs, additional_inputs):
    return generic_associative_scan(combine_fn, xs, additional_inputs=additional_inputs)


def _leaf_dependencies(
    combine_fn_bw_gm: torch.fx.GraphModule, num_xs: int, num_additional_inputs: int
) -> tuple[list[list[bool]], list[list[int]], list[bool]]:
    """
    Reads the dependencies between the xs leaves off the joint graph of one combine
    step, see NOTE: [associative_scan leaf dependencies]. Returns:

    deps[i][j]: whether the upstream gradient of output leaf i reaches the gradient
        of the previous output ys{t-1}^j for j < num_xs, or of the input
        xst^(j - num_xs) otherwise.
    groups: the leaves grouped such that within a group the previous outputs depend
        on each other, ordered such that gradients only flow into later groups.
    elementwise: for each group, whether its outputs depend on the previous outputs
        of the group through pointwise operations and layout changes only.
    """
    graph = combine_fn_bw_gm.graph
    tangents = graph.find_nodes(op="placeholder")[2 * num_xs + num_additional_inputs :]
    outputs = graph.output_node().args[0]
    if not isinstance(outputs, (list, tuple)):
        raise AssertionError(f"Expected the joint to return a list, got {outputs}")
    grads = outputs[: 2 * num_xs]

    def closure(start, step):
        seen, todo = set(), list(start)
        while todo:
            n = todo.pop()
            if n not in seen:
                seen.add(n)
                todo.extend(step(n))
        return seen

    below = [closure([t], lambda n: n.users) for t in tangents]
    deps = [[g in b for g in grads] for b in below]

    # reach[i]: the leaves whose g_ys receives contributions from the g_ys of leaf i
    reach = [
        closure([i], lambda k: [j for j in range(num_xs) if deps[k][j]])
        for i in range(num_xs)
    ]
    groups: list[list[int]] = []
    for i in range(num_xs):
        if not any(i in group for group in groups):
            groups.append(sorted(j for j in reach[i] if i in reach[j]))
    # Every leaf that reaches a group also reaches the groups it flows into
    groups.sort(key=lambda group: sum(group[0] in r for r in reach))

    def numel(n):
        val = n.meta.get("val")
        return val.numel() if isinstance(val, torch.Tensor) else None

    def is_elementwise(n, on_paths):
        if (
            n.op != "call_function"
            or n.target is operator.getitem
            or torch.Tag.pointwise in getattr(n.target, "tags", ())
            or numel(n) is None
        ):
            return True
        # Layout changes, such as the permutes of vmap, keep every element, while
        # indexing or broadcasting a slice changes the number of elements
        return all(
            statically_known_true(numel(a) == numel(n))
            for a in n.all_input_nodes
            if a in on_paths and numel(a) is not None
        )

    elementwise = []
    for group in groups:
        above = closure([grads[j] for j in group], lambda n: n.all_input_nodes)
        on_paths = above & set().union(*[below[i] for i in group])
        elementwise.append(all(is_elementwise(n, on_paths) for n in on_paths))

    return deps, groups, elementwise


class AssociativeScanAutogradOp(torch.autograd.Function):
    r""" associative_scan
        Example::
            xs = torch.arange(1, 5) = [1, 2, 3, 4]

            def combine_fn(a: torch.Tensor, b: torch.Tensor):
                return a * b

            ys = associative_scan(comine_fn, xs),
            which can be unpacked as:
            ys0 = xs0                                         = 1
            ys1 = combine_fn(ys0, xs1) = combine_fn(1, 2)     = 2
            ...
            ysT = combine_fn(ys(T-1), xsT) = combine_fn(6, 4) = 24
            ys = [1, 2, 6, 24]

            This creates a recursive data dependency structure where each output yst
            depends on all prior inputs xs0 through xst. The dependency can be visualized as:

        Level 0 (Input):    xs0      xs1      xs2      xs3      xs4
                              \      /          |        |        |
                               \    /           |        |        |
        Level 1:                ys1 ────────────┘        |        |
                                  \                     /         |
                                   \                   /          |
        Level 2:                    ys2 ───────────────┘          |
                                      \                          /
                                       \                        /
        Level 3:                        ys3 ────────────────────┘
                                          \
        Level 4:                           ys4


        We could get the following backward gradient graph:


        Level 0 (output):   g_xs0    g_xs1    g_xs2    g_xs3    g_xs4
                              \      /          |        |        |
                               \    /           |        |        |
        Level 1:    gl_ys1 ──> g_ys1 ───────────┘        |        |
                                  \                     /         |
                                   \                   /          |
        Level 2:    gl_ys2 ──> g_ys2 ───────────────────┘         |
                                      \                          /
                                       \                        /
        Level 3:    gl_ys3 ──> g_ys3 ────────────────────────────┘
                                          \
        Level 4:    gl_ys4 ──> g_ys4

        where gl_y1 is the gradient of the loss with respect to ys1 and the input of backward.

        To calculate the gradients of the inputs, the chain rule suggests:

        g_xs0 = g_ys1
        g_xs1 = g_ys1 * bw(ys0, xs1) = g_ys1 * bwxs01
        g_xs2 = g_ys2 * bw(ys1, xs2) = g_ys2 * bwxs12
        g_xs3 = g_ys3 * bw(ys2, xs3) = g_ys3 * bwxs23
        g_xs4 = g_ys4 * bw(ys3, xs4) = g_ys4 * bwxs34

        Notice the bw(...) is just the single step bw (instantaneous gradients), whose formula can be computed from combine_fn.
        For example bw(ys3, xs4) (also abbreviated with bwxs34) computes the gradients ∂/∂xs4 combine_fn(ys3, xs4).
        Similarly, bw(ys4, ys3) (also abbreviated with bwys43) computes the gradients ∂/∂ys3 combine_fn(ys3, xs4).

        Let's break down how to calculate g_ys by recursively substituting the unknowns:

        g_ys1 = gl_ys1 + g_ys2 * bw(ys2, ys1)
              = gl_ys1 + (gl_ys2  + g_ys3 * bw(ys3, ys2)) * bw(ys2, ys1)
              = gl_ys1 + gl_ys2 * bw(ys2, ys1) + g_ys3 * bw(ys3, ys2) * bw(y2, y1)
              = gl_ys1 + gl_ys2 * bw(ys2, ys1) + gl_ys3 * bw(ys3, ys2) * bw(y2, y1)
                    + gl_ys4 * bw(ys4, ys3) * bw(ys3, ys2) * bw(ys2, ys1)

        Let's do the same for all the g_ys:
        g_ys2 = gl_ys2 + gl_ys3 * bw(ys3, ys2) + gl_ys4 * bw(ys4, ys3) * bw(ys3, ys2)
        g_ys3 = gl_ys3 + gl_ys4 * bw(ys4, ys3)
        g_ys4 = gl_ys4

        Notice that the above can be re-written as a right-to-left associative_scan with flat operator

        def g_ys_combine_fn_flat(bw, gl, bw_next, gl_next):
            return bw * bw_next, torch.addcmul(gl_next, tensor1=bw_next, tensor2=gl) # gl_next + bw_next * gl

        The recurrence g_yst = gl_yst + g_ys{t+1} * bw(ys{t+1}, yst) means step t consumes
        bwys shifted one step forward: bwys_aligned = [bwys21, bwys32, bwys43, 1], where
        bwys21 abbreviates bw(ys2, ys1) and so on. The final step (g_ys4) has no successor,
        so bwys_aligned is padded with a single 1. gl_ys is used as-is (no padding):

        bwys_aligned = [bwys21, bwys32, bwys43, 1]
        gl_ys        = [gl_ys1, gl_ys2, gl_ys3, gl_ys4]

        g_ys is recovered by flipping, scanning left-to-right, and flipping back:
            leaves_rev = [bwys_aligned.flip([0]), gl_ys.flip([0])]
            result_rev = associative_scan_op(g_ys_combine_fn_flat, leaves_rev, ())
            g_ys = result_rev[1].flip([0])

        With N xs leaves, one step maps the N-tuples ys{t-1} and xst to the N-tuple yst,
        and every output leaf may depend on every input leaf. The local gradients
        bw(ys{t+1}, yst) and bw(yst, xst) then become N x N Jacobians over the leaf
        index, whose entries are tensors of the leaf shape, see
        NOTE: [associative_scan leaf dependencies]. The chain rule sums over the leaves:

        g_yst^i = gl_yst^i + sum_j g_ys{t+1}^j * bw(ys{t+1}^j, yst^i)
        g_xst^i = sum_j g_yst^j * bw(yst^j, xst^i)

        The same scan applies if the argument bw of g_ys_combine_fn_flat holds the
        transposed Jacobian, with bw(ys{t+1}^j, yst^i) in row i and column j, and gl
        holds a vector: bw * bw_next becomes the matrix product bw_next @ bw,
        bw_next * gl becomes the matrix-vector product bw_next @ gl, and the pad becomes
        the identity matrix. For N = 1 this is the scalar case above.

        References: https://justintchiu.com/blog/pscan_diff/

        NOTE: [associative_scan autograd implementation]

        The forward of associative_scan can be computed with the following steps:

        1.) Compute the forward output of the associative_scan
            ys = associative_scan(combine_fn, xs, additional_inputs)

        The backward of associative_scan can be computed with the following steps:

        2.) Prepare the backward graph
            We prepare the backward graph to be used in the backward function.
            We utilize ``create_bw_fn`` to generate the joint function:
            combine_fn_bw = create_bw_fn(combine_fn, operands)
            where operands = [ys{t-1}, xst, additional_inputs]

        3.) Materialize the ``combine_fn_bw``
            This is required because torch.compile and torch.autograd.grad
            cannot trace through the joint backward function dynamically.

        4.) Compute the single step bw (instantaneous gradients) at every step t
            bwys[i] = combine_fn_bw(ys{t-1}, xst, e_i)
            Here we pass e_i, i.e. 1 for output leaf i and 0 for all other leaves, as
            the upstream gradient to obtain row i of the local ys-Jacobian.

            This gives, for every pair of leaves (i, j):
                bwys[i][j] = [bw(ys1^i, ys0^j), bw(ys2^i, ys1^j), ..., bw(ysT^i, ys{T-1}^j)]

        5.) Compute the gradients using a right-to-left associative scan

            As shown in the example above, each input xst affects all later outputs ysi for i >= t.
            According to the chain rule, each such path contributes a product of local gradients g_ysk.

            For example:
                g_yst = gl_yst + g_ys{t+1} * bw(ys{t+1}, yst)

            This motivates a right-to-left associative scan over bwys and gl_ys.
            We call the raw associative_scan_op HOP with the flat operator
            g_ys_combine_fn_flat above, so the backward scan is Triton-lowerable under
            compiled autograd; in eager it dispatches to generic_associative_scan.

            5.1) Align bwys to the recurrence

                bwys_aligned[i][j] = torch.cat([bwys[j][i][1:], identity[i][j]], 0)

                Step t consumes the transpose of bwys[t+1] (the local ys-Jacobian of the
                next step), so bwys is shifted one step forward. The final step has no
                successor, so the identity is appended as padding; it does not affect
                the final g_ys. gl_ys is used unchanged.

            5.2) Flip, scan left-to-right, and flip back

                leaves_rev = [*bwys_aligned.flip([0]), *gl_ys.flip([0])]
                result_rev = associative_scan_op(g_ys_combine_fn_flat, leaves_rev, ())
                g_ys = result_rev[N * N:].flip([0])

                where bwys_aligned is flattened in row-major order.

        6.) Contract g_ys with the instantaneous input gradients, see g_xst above
            Since combine_fn_bw is a VJP, this is one more invocation of it, seeded
            with g_ys instead of a basis vector, so the xs-Jacobian is not materialized.
            The first step is excluded, as ys0 equals xs0 and no combine_fn is applied.

            This gives the final input gradients:
                g_xs = [∂L/∂xs0, ∂L/∂xs1, ..., ∂L/∂xsT]

        NOTE: [associative_scan leaf dependencies]
            The scan of 5.) multiplies the entries of the ys-Jacobian elementwise. This
            requires that each element of an output leaf only depends on the same
            element of the previous outputs it is coupled with, and that coupled leaves
            have the same shape. Therefore, the leaves are grouped by their dependencies,
            which _leaf_dependencies reads off the graph of combine_fn_bw: a group holds
            the leaves whose previous outputs depend on each other, possibly through
            other leaves. Steps 4.) and 5.) are performed per group. Across groups,
            gradients flow in one direction only, e.g. from the states to a decay factor
            that they read. The groups are processed in that direction, and the g_ys of
            the already processed leaves j enter a group through

                gl_yst^i += sum_j g_ys{t+1}^j * bw(ys{t+1}^j, yst^i)

            which is one more invocation of combine_fn_bw, seeded with those g_ys. Like
            the contraction of 6.), it handles any operation between the groups, such
            as broadcasting or indexing.

            Groups whose g_ys does not reach any xs leaf that requires gradients are
            skipped. For the other groups, the backward raises if coupled leaves differ
            in shape or depend on each other through operations that change the number
            of elements, such as indexing, and for complex leaves, as the basis vectors
            of 4.) only give the Jacobian of a holomorphic combine_fn. Permutations of
            the elements within a group are not detected. Coupled leaves of different real dtypes are
            processed in their promoted dtype.

        NOTE: [cost of the leaf-index Jacobian]
            A group of n coupled leaves carries n * n Jacobian entries through the scan
            of 5.) and performs O(n**3) elementwise multiply-adds per combine step.
            Jacobian entries that are structurally zero within a group are not skipped.
            The basis vectors of leaves in different groups whose Jacobian rows do not
            overlap share one invocation of combine_fn_bw in 4.), so N independent
            leaves need a single one, as for N = 1.

        NOTE: [scan partial grad handling]
            If any element of xs or of the outputs does not require gradients
            (i.e., requires_grad=False), then the corresponding gradients will be returned
            as tensors of zeros with the same shape as the element.
    """

    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(
        ctx,
        combine_fn,
        num_xs,
        num_additional_inputs,
        *operands,
    ):
        ctx._num_xs = num_xs
        ctx._num_additional_inputs = num_additional_inputs
        ctx._combine_fn = combine_fn
        xs, additional_inputs = split_into_chunks(
            operands, [num_xs, num_additional_inputs]
        )

        scan_length = xs[0].shape[0]
        ctx._scan_length = scan_length

        # We snapshot the dispatch keys in forward for materializing
        # the bw_graph in backward.
        ctx._fw_include_key_set = torch._C._dispatch_tls_local_include_set()
        ctx._fw_exclude_key_set = torch._C._dispatch_tls_local_exclude_set()

        with torch._C._AutoDispatchBelowAutograd():
            # 1.) Compute the forward output of the associative_scan
            ys = associative_scan_op(combine_fn, xs, additional_inputs)
            save_values_for_backward(ctx, list(operands) + list(ys))

        return (*ys,)

    @staticmethod
    def backward(ctx, *gl_ys):
        r"""
        This function computes the gradients of the scan operation.
        For a detailed description see the document above.

        Args:
            flat_grads (torch.Tensor): The tensor of upstream gradients, or a nested pytree of tensors.
                                       E.g.: Gradient of the loss with respect to the forward output ys
        """

        # The backward of associative_scan is always performed on the first dimension
        dim = 0
        scan_length = ctx._scan_length
        num_xs = ctx._num_xs
        num_additional_inputs = ctx._num_additional_inputs

        # Extract the inputs to the forward path and outputs from the forward path
        flat_args = saved_values(ctx)
        xs, additional_inputs, outs = split_into_chunks(
            flat_args, [num_xs, num_additional_inputs, num_xs]
        )

        if scan_length == 0:
            return (
                *[None] * 3,
                *[torch.zeros_like(x) for x in xs],
                *[None] * num_additional_inputs,
            )

        # First_slice_copy does not keep the original requires_grad flag,
        # but we need it here in order to compute the correcte gradients.
        # The previous outputs ys{t-1} require grad even if their xs leaf does not,
        # because other leaves can depend on them.
        xs_slices = first_slice_copy_with_grad(itertools.chain(outs, xs))

        # Construct the operands from the forward, fw_operands
        # and the operands for a single event t of the forward, fw_operands_slice
        fw_operands = (*xs, *additional_inputs)
        fw_operands_slice = (*xs_slices, *additional_inputs)

        # 2.) Prepare the backward graph
        combine_fn_bw = create_bw_fn(ctx._combine_fn, fw_operands_slice)

        # 3.) Materialize the ``combine_fn_bw``
        # TODO: we need to materialize the bw graphs because dynamo is unable to
        # trace through the joint function when torch.compile torch.autograd.grad.
        combine_fn_bw_gm = materialize_as_graph(
            combine_fn_bw,
            (
                *fw_operands_slice,
                *[first_slice_copy(o) for o in outs],
            ),
            ctx._fw_include_key_set,
            ctx._fw_exclude_key_set,
            force_enable_grad=True,
        )

        # vmap joint graph over scan dimension to compute the individual
        # gradients for each time slice ``t`` in parallel.
        # This computation can be parallelized, as these are just the instantaneous gradients and not the full chain-rule
        # Only the per-timestep tensors are mapped over dim 0; additional_inputs are
        # lifted (not stacked along the scan dim), so they are broadcast via in_dim=None.
        in_dims = (
            *([0] * num_xs),
            *([0] * num_xs),
            *([None] * num_additional_inputs),
            *([0] * num_xs),
        )

        # create_bw_fn returns one grad per primal: 2*num_xs (for ys{t-1} and xs) plus
        # one per additional_input, appended last. Drop the trailing additional_input grads
        # as they are not differentiated.
        def combine_fn_bw_gm_xs(*args):
            return combine_fn_bw_gm(*args)[: 2 * num_xs]

        # pyrefly: ignore [bad-argument-type]
        mapped_combine_fn_bw_gm = torch.vmap(combine_fn_bw_gm_xs, in_dims, 0)

        # 4.) to 6.) evaluate the vmapped joint at the previous outputs ys{t-1}
        prev_outs = [o.roll(1, dim) for o in outs]

        def step_bw(upstream_grads):
            return mapped_combine_fn_bw_gm(*prev_outs, *fw_operands, *upstream_grads)

        # Entry (i, j) of the identity matrix over the leaf index, shaped like ``t``
        def eye_like(i: int, j: int, t: torch.Tensor) -> torch.Tensor:
            return torch.ones_like(t) if i == j else torch.zeros_like(t)

        def compute_gys_associative_scan(
            gl_ys: list[torch.Tensor], bwys: list[list[torch.Tensor]]
        ) -> list[torch.Tensor]:
            """
            Computes the gradient g_ys via a right-to-left associative scan:
            I.e., the gradients are computed from the last time step to the first, following this equation
                g_yst^i = gl_yst^i + sum_j g_ys{t+1}^j * bw(ys{t+1}^j, yst^i)
            """
            n = len(gl_ys)
            idx = range(n)

            # 5.1) Align bwys to the recurrence above: step t consumes the transpose of
            # bwys[t+1], and the last step (t = T-1) has no successor, so pad with the
            # identity.
            bwys_aligned = [
                torch.cat([bwys[j][i][1:], eye_like(i, j, bwys[j][i][0:1])], 0)
                for i in idx
                for j in idx
            ]

            def dot(pairs, acc=None):
                # sum(a * b for a, b in pairs), accumulated with addcmul onto ``acc``
                for a, b in pairs:
                    acc = a * b if acc is None else torch.addcmul(acc, a, b)
                return acc

            def g_ys_combine_fn_flat(*args):
                # An element is the affine map v -> bw @ v + gl, flattened into the
                # n * n entries of bw in row-major order and the n entries of gl.
                bw, gl, bw_next, gl_next = split_into_chunks(args, [n * n, n, n * n, n])
                bw = split_into_chunks(bw, [n] * n)
                bw_next = split_into_chunks(bw_next, [n] * n)

                # bw_next @ bw and gl_next + bw_next @ gl
                new_bw = [
                    dot((bw_next[i][k], bw[k][j]) for k in idx)
                    for i in idx
                    for j in idx
                ]
                new_gl = [
                    dot(((bw_next[i][k], gl[k]) for k in idx), gl_next[i]) for i in idx
                ]
                # A list, as dynamo requires the same pytree structure as the xs below
                return [*new_bw, *new_gl]

            # 5.2) Flip, scan left-to-right, and flip back to get g_ys. We call the raw
            # associative_scan_op HOP (not generic_associative_scan) so this scan is
            # Triton-lowerable under compiled autograd.
            result_rev = associative_scan_op(
                g_ys_combine_fn_flat,
                [x.flip([0]) for x in (*bwys_aligned, *gl_ys)],
                (),
            )
            g_ys = [r.flip([0]) for r in result_rev[n * n :]]

            return g_ys

        # See NOTE: [associative_scan leaf dependencies]
        deps, groups, elementwise = _leaf_dependencies(
            combine_fn_bw_gm, num_xs, num_additional_inputs
        )
        group_of = {i: group for group in groups for i in group}

        # The leaves whose g_ys reaches the gradient of an xs leaf that requires it
        needs_grad = ctx.needs_input_grad[3 : 3 + num_xs]
        needed: set[int] = set()
        for group in reversed(groups):
            if any(
                needs_grad[j]
                or any(deps[j][num_xs + k] and needs_grad[k] for k in range(num_xs))
                or any(deps[j][k] for k in needed)
                for j in group
            ):
                needed.update(group)

        def unsupported(reason: str) -> RuntimeError:
            return RuntimeError(
                "associative_scan with combine_mode='pointwise' does not support "
                f"gradients for {reason}, consider combine_mode='generic'"
            )

        for group, is_elementwise in zip(groups, elementwise):
            if group[0] not in needed:
                continue
            if any(xs[j].shape[1:] != xs[group[0]].shape[1:] for j in group):
                raise unsupported(
                    "xs leaves of different shapes whose previous outputs depend on "
                    "each other"
                )
            if any(xs[j].is_complex() for j in group):
                raise unsupported("complex xs leaves")
            if not is_elementwise:
                raise unsupported(
                    "outputs that depend on previous outputs through indexing or other "
                    "operations that change the number of elements"
                )

        # 4.) Compute the single step bw (instantaneous gradients) at every step: the
        # rows of the local ys-Jacobian within each group. The basis vectors of leaves
        # whose rows do not overlap share one invocation.
        bwys: dict[int, list[torch.Tensor]] = {}
        todo = [i for i in range(num_xs) if i in needed]
        while todo:
            batch: list[int] = []
            for i in todo:
                if all(
                    group_of[i] is not group_of[k]
                    and not any(deps[i][j] for j in group_of[k])
                    and not any(deps[k][j] for j in group_of[i])
                    for k in batch
                ):
                    batch.append(i)
            bw_batch = step_bw(
                [
                    (torch.ones_like if j in batch else torch.zeros_like)(gl)
                    for j, gl in enumerate(gl_ys)
                ]
            )
            for i in batch:
                bwys[i] = [bw_batch[j] for j in group_of[i]]
            todo = [i for i in todo if i not in batch]

        g_ys = [torch.zeros_like(gl) for gl in gl_ys]
        for group in groups:
            if group[0] not in needed:
                continue
            dtype = functools.reduce(torch.promote_types, (xs[j].dtype for j in group))

            # The g_ys of earlier groups enter through the previous outputs of this group
            # that they read. The g_ys of this and later groups are still zero.
            gl_group = [gl_ys[j] for j in group]
            if any(deps[k][j] for k in range(num_xs) if k not in group for j in group):
                bw_earlier = step_bw(g_ys)
                # As in 5.1), step t contributes to g_ys{t-1}
                gl_group = [
                    gl + torch.cat([bw_earlier[j][1:], torch.zeros_like(gl[0:1])], 0)
                    for gl, j in zip(gl_group, group)
                ]

            # 5.) Compute the gradients via an associative_scan
            g_ys_group = compute_gys_associative_scan(
                [gl.to(dtype) for gl in gl_group],
                [[b.to(dtype) for b in bwys[i]] for i in group],
            )
            for j, g in zip(group, g_ys_group):
                g_ys[j] = g.to(xs[j].dtype)

        # 6.) Contract with the instantaneous input gradients
        g_xs = step_bw(g_ys)[num_xs:]
        # The first output ys0 equals xs0, so no combine_fn is applied at t = 0
        g_xs = [torch.cat([g[0:1], gx[1:]], 0) for g, gx in zip(g_ys, g_xs)]

        # TODO: Currently the gradients for the additional_inputs are not computed properly
        return *[None] * 3, *g_xs, *[None] * num_additional_inputs


@associative_scan_op.py_autograd_impl
def associative_scan_autograd(combine_fn, xs, additional_inputs):
    num_xs = len(xs)

    # additional_inputs may interleave Tensors with integer/SymInt constants lifted
    # by dynamo (e.g. shape SymInts of a dynamic-shaped closed-over tensor, inserted
    # before the tensor itself). Only Tensor additional_inputs participate in autograd;
    # gradients for lifted parameters are not supported yet.
    # NOTE: the isinstance guard below is defensive against such interleaved non-Tensor
    # entries. It is currently unexercised in CI because every pointwise autograd test
    # skips compile_dynamic_shape, which is what would produce a lifted SymInt here.
    if any(a.requires_grad for a in additional_inputs if isinstance(a, torch.Tensor)):
        raise RuntimeError(
            "Associative_scan does currently not support gradients for lifted parameters!"
        )

    # Pass all additional_inputs through in their original order. combine_fn is invoked
    # purely positionally as operator(*lhs, *rhs, *additional_inputs), so preserving the
    # interleaved order is required; backward excludes them from the vmap batch dims and
    # drops their grad slots (see AssociativeScanAutogradOp.backward).
    flat_out = AssociativeScanAutogradOp.apply(
        combine_fn,
        num_xs,
        len(additional_inputs),
        *(tuple(xs) + tuple(additional_inputs)),
    )
    return (*flat_out,)


@associative_scan_op.py_impl(ProxyTorchDispatchMode)
def associative_scan_proxy_mode(mode, combine_fn, xs, additional_inputs):
    return trace_associative_scan(
        mode, associative_scan_op, combine_fn, xs, additional_inputs
    )


@register_fake(associative_scan_op, skip_cache=True)
def assoiciative_scan_fake_tensor_mode(combine_fn, xs, additional_inputs):
    return tuple(x.clone() for x in xs)


@associative_scan_op.py_functionalize_impl
def associative_scan_functionalize(ctx, combine_fn, xs, additional_inputs):
    from torch._higher_order_ops.utils import _check_alias_and_mutation

    unwrapped_xs = ctx.unwrap_tensors(xs)
    unwrapped_additional_inputs = ctx.unwrap_tensors(additional_inputs)
    with ctx.redispatch_to_next():
        functional_combine_fn = ctx.functionalize(
            _maybe_run_with_interpreter(combine_fn)
        )
        pre_dispatch = hasattr(ctx, "mode") and ctx.mode.pre_dispatch
        sample_unwrapped_xs_sliced = [
            first_slice_copy(inp) for inp in itertools.chain(unwrapped_xs, unwrapped_xs)
        ]
        sample_inputs = list(
            itertools.chain(
                sample_unwrapped_xs_sliced,
                unwrapped_additional_inputs,
            )
        )
        _check_alias_and_mutation(
            combine_fn, sample_inputs, "associative_scan", pre_dispatch
        )
        ret = associative_scan_op(
            functional_combine_fn,
            unwrapped_xs,
            unwrapped_additional_inputs,
        )
    return ctx.wrap_tensors(ret)


# Note [associative_scan vmap coverage]
# This batch rule is dispatched only when the associative_scan_op HOP is present
# under a vmap layer. The frontend builds that HOP for combine_mode="pointwise";
# combine_mode="generic" is a pure-Python decomposition (generic_associative_scan)
# that never constructs the HOP, so vmap over a generic scan is handled entirely
# by the batching rules of the individual aten ops and never reaches this rule.
# Consequently only the pointwise cases in the vmap tests exercise the code below;
# the generic cases guard the frontend decomposition instead. This rule is itself
# device-agnostic, since in eager the HOP dense-decomposes on any device; the
# CUDA-only parametrization of those tests and the compile-time lowering failure
# are properties of the lowered pointwise scan, not of this rule.
@associative_scan_op.py_impl(torch._C._functorch.TransformType.Vmap)
def associative_scan_batch_rule(interpreter, combine_fn, xs, additional_inputs):
    unbatched_args, in_dims = unwrap_batched(
        (xs, additional_inputs), interpreter.level()
    )
    # move to last dim to not interfere with scan's batching
    unbatched_xs, unbatched_additional_inputs = _move_batch_dims_to_last_for_scan(
        unbatched_args, in_dims
    )
    xs_in_dims, additional_in_dims = in_dims
    xs_move_dims = _batch_dims_as_last_for_scan(xs_in_dims)
    additional_move_dims = _batch_dims_as_last_for_scan(additional_in_dims)
    # combine_fn is called with (lhs xs leaves, rhs xs leaves, additional_inputs),
    # so the xs batch-dim markers must be duplicated. See generic_associative_scan.
    after_move_dims = (*xs_move_dims, *xs_move_dims, *additional_move_dims)

    with interpreter.lower():
        # generic_associative_scan feeds combine_fn outputs back as the left-hand
        # args on later levels, reusing after_move_dims; that is only valid if the
        # outputs keep the same batch dims as xs. expected_out_dims makes the wrapper
        # raise a clear error otherwise instead of silently mismatching downstream.
        wrapper = _VmapCombineFnWrapper(
            combine_fn,
            after_move_dims,
            interpreter.batch_size(),
            interpreter.randomness(),
            expected_out_dims=xs_move_dims,
            op_name="associative_scan",
        )
        unwrapped_out = associative_scan_op(
            wrapper, unbatched_xs, unbatched_additional_inputs
        )

    out_dims = wrapper.out_dims
    if out_dims is None:
        # Scan was a no-op (scan length < 2): the combine_fn is never called, so
        # outputs alias xs one-to-one and their batch dims equal the xs batch dims.
        out_dims = xs_move_dims
    # wrap_batched matches bdims against the output container; associative_scan_op
    # returns a list, so pass a tuple to align with the tuple out_dims.
    return wrap_batched(tuple(unwrapped_out), out_dims, interpreter.level())


def _fake_associative_scan(combine_fn, xs, dim, reverse=False):
    inp_leaves, spec = pytree.tree_flatten(xs)
    result_flat: list[Any] = []
    num_leaves = len(inp_leaves)
    op = reversed if reverse else lambda x: x

    for ind in op(range(inp_leaves[0].size(dim))):
        r = [
            inp_leaves[leave_ind][(slice(None),) * dim + (ind,)]
            for leave_ind in range(num_leaves)
        ]
        if (ind > 0 and not reverse) or (
            ind < (inp_leaves[0].size(dim) - 1) and reverse
        ):
            r = combine_fn(
                pytree.tree_unflatten(result_flat[-1], spec),
                pytree.tree_unflatten(r, spec),
            )
        r_flat, _ = pytree.tree_flatten(r)
        result_flat.append(r_flat)

    if len(result_flat) == 0:
        results = list(inp_leaves)
    else:
        results = [
            torch.stack([e[leave_ind] for e in op(result_flat)], dim)
            for leave_ind in range(num_leaves)
        ]
    return pytree.tree_unflatten(results, spec)
