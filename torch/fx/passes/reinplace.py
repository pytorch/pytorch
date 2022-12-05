import torch
from torch.fx import Node
from torch.fx._compatibility import compatibility
from torch._subclasses.fake_tensor import FakeTensorMode, FakeTensor
from torch.utils._pytree import tree_map, tree_flatten, tree_map_only
from torch.multiprocessing.reductions import StorageWeakRef

import _operator
from enum import Enum
import itertools
from typing import Set, Dict
from collections import defaultdict

__all__ = ['reinplace']

class _ViewType(Enum):
    NonView = 0
    SingleOutputView = 1
    MultiOutputView = 2

def _is_view_op(tgt):
    if tgt is not None and isinstance(tgt, torch._ops.OpOverload):
        schema = tgt._schema
        if len(schema.arguments) > 0:
            first_arg = schema.arguments[0]
            # check if op is a view
            return first_arg.alias_info is not None and not first_arg.alias_info.is_write

def _get_view_type(tgt) -> _ViewType:
    if tgt is not None and isinstance(tgt, torch._ops.OpOverload):
        schema = tgt._schema
        if len(schema.arguments) > 0:
            first_arg = schema.arguments[0]
            # check if op is a view
            if first_arg.alias_info is not None and not first_arg.alias_info.is_write:
                # check if op is a multi-output view
                if '*' in first_arg.alias_info.after_set:
                    return _ViewType.MultiOutputView
                else:
                    return _ViewType.SingleOutputView
    return _ViewType.NonView


# Stores a bunch of metadata related to functionalization each node.
# Relevant metadata:
# n.meta['fake_result']: FakeTensor (same type as the output of the node, but with FakeTenors instead of Tensors)
#   The fake tensor output from running the current node
# n.meta['view_of']: Node
#   If the current node n is a view of some base tensor, the 'view_of' field tells us which
#   view node was used to generate the current node (a view tensor).
#   This information actually makes `fake_result` redundant, but we can use `fake_result`
#   to sanity check that our aliasing information is correct.
@compatibility(is_backward_compatible=False)
class _FunctionalizationMetadataProp(torch.fx.Interpreter):

    def run_node(self, node: Node):
        self.node_counter += 1
        result = super().run_node(node)
        node.meta['fake_result'] = result
        node.meta['node_idx'] = self.node_counter

        # (1) Update metadata with the list of nodes that are used by this node
        # copy_() doesn't read from its first argument; it writes to it, overwriting previous data.
        # We don't want to treat it as "being used as an input".
        node_args = node.args
        if node.target is torch.ops.aten.copy_.default:
            node_args = node_args[1:]

        # (2) Update metadata to track aliasing information about view tensor nodes.
        if node.op == 'call_function':
            view_type = _get_view_type(node.target)
            if view_type == _ViewType.SingleOutputView:
                assert isinstance(node.args[0], Node)
                node.meta['view_of'] = node.args[0]
            elif view_type == _ViewType.MultiOutputView:
                self.multi_output_view_nodes[node] = node.args[0]

            # Check if we returned a multi-output view,
            # and we're now grabbing the individual views from the output.
            #
            # For multi-output views, we want to map each output view to the base,
            # but this mapping involves two separate nodes in FX IR.
            # e.g. "a, b = x_1.split(...)" becomes:
            #    %split_tensor : [#users=2] = call_function[target=torch.ops.aten.split.Tensor](args = (%x_1, 2), kwargs = {})
            #    %getitem : [#users=1] = call_function[target=operator.getitem](args = (%split_tensor, 0), kwargs = {})
            #    %getitem_1 : [#users=1] = call_function[target=operator.getitem](args = (%split_tensor, 1), kwargs = {})
            # And we'd like to set:
            #    getitem1.meta['view_of'] = x_1
            elif node.target is _operator.getitem:
                list_arg = node.args[0]
                maybe_base_of_view = self.multi_output_view_nodes.get(list_arg, None)
                if maybe_base_of_view is not None:
                    # Note: we could also track indexing info here for multi-output views.
                    # I don't think this metadata is strictly needed for de-functionalization.
                    assert isinstance(maybe_base_of_view, Node)
                    node.meta['view_of'] = maybe_base_of_view

        if 'view_of' in node.meta:
            # We're linking the current node with its first argument as views.
            # Assert here that this is actually the case, and their storages are the same.
            assert isinstance(node.meta['fake_result'], FakeTensor)
            assert isinstance(node.meta['view_of'].meta['fake_result'], FakeTensor)
            view_storage = StorageWeakRef(node.meta['fake_result']._typed_storage())
            base_storage = StorageWeakRef(node.meta['view_of'].meta['fake_result']._typed_storage())
            assert view_storage == base_storage
        return result



    def propagate(self, *args):
        self.multi_output_view_nodes = {}
        self.node_counter = -1

        with FakeTensorMode(allow_meta=True) as mode:
            fake_args = [mode.from_tensor(a) for a in args]
            return super().run(*fake_args)

def _schemas_match(functional_schema, inplace_schema):
    names_match = inplace_schema.name.endswith("_") and inplace_schema.name[:-1] == functional_schema.name
    arg_types_match = len(functional_schema.arguments) == len(inplace_schema.arguments) and all(
        a1.type == a2.type for a1, a2 in zip(functional_schema.arguments, inplace_schema.arguments))
    # for the inplace op, its first argument should be mutable
    assert inplace_schema.arguments[0].alias_info is not None and inplace_schema.arguments[0].alias_info.is_write
    # and its remaining arguments shouldn't be.
    assert all(a.alias_info is None for a in inplace_schema.arguments[1:])
    return names_match and arg_types_match

# TODO: this should be beefed up to be able to properly re-inplace with:
# - mutating ops (e.g. _fused_moving_avg_obs_fq_helper)
# - out= ops (e.g. angle -> angle.out)
# TODO: we should also figure this info out using torchgen.
def _maybe_get_inplace_op(op):
    # __module__ seems broken; it returns torch._ops.aten which doesn't exist
    if not isinstance(op, torch._ops.OpOverload):
        return None
    # Some view ops have inplace variants (as_strided_, etc),
    # but we do NOT want the reinplacing pass to directly add these into the program.
    # (they'll require extra special handling, aren't aren't really useful for perf anyway)
    if _is_view_op(op):
        return None
    op_namespace = op.__module__.split(".")[-1]
    op_base_name = op.overloadpacket.__name__
    maybe_namespace_module = getattr(torch.ops, op_namespace)
    maybe_inplace_op = None if maybe_namespace_module is None else getattr(maybe_namespace_module, f'{op_base_name}_', None)
    if maybe_inplace_op is None:
        return None

    inplace_overloads = [
        getattr(maybe_inplace_op, overload_name) for overload_name in maybe_inplace_op.overloads()
    ]
    inplace_overloads_with_matching_schemas = [
        f
        for f in inplace_overloads
        if _schemas_match(op._schema, f._schema)
    ]
    # Just becuase foo() and foo_() are both existing operators,
    # They aren't guaranteed to have compatible schemas.
    # For example, pow.Scalar(Scalar self, Tensor exponent) has no valid inplace variant,
    # Even though several overloads of pow_ exist.
    if len(inplace_overloads_with_matching_schemas) == 0:
        return None
    assert len(inplace_overloads_with_matching_schemas) == 1
    inplace_op = inplace_overloads_with_matching_schemas[0]
    return inplace_op

_VIEW_INVERSE_MAP = {
    torch.ops.aten.diagonal_scatter.default: torch.ops.aten.diagonal.default,
    torch.ops.aten.select_scatter.default: torch.ops.aten.select.int,
    torch.ops.aten.slice_scatter.default: torch.ops.aten.slice.Tensor,
    torch.ops.aten.as_strided_scatter.default: torch.ops.aten.as_strided.default,
}

# This function, given a set of set of (aliased) tensor nodes,
# Returns any nodes in the graph that *use* any of the aliases, that occur *after* op_index
# in the node ordering.
def _get_all_later_node_usages(tensor_aliases: Set[Node], op_index: int):
    def _add_if_tensor(x, set_):
        if isinstance(x, FakeTensor):
            set_.add(StorageWeakRef(x._typed_storage()))

    nodes_used_after = set()
    for t in tensor_aliases:
        # get all nodes that use the current alias
        usage_nodes = t.users
        for n in usage_nodes:
            # We only care about usages after the current node
            if 'node_idx' not in n.meta or n.meta['node_idx'] <= op_index:
                continue
            # We also don't care about intermediate view ops.
            # They only matter if their output is then used elsewhere
            # (either in an out-of-place op, or as an output to the function).
            if n in tensor_aliases:
                if isinstance(n.target, torch._ops.OpOverload) or n.target == _operator.getitem:
                    continue
            nodes_used_after.add(n)
    return nodes_used_after

# Given an op that we're trying to re-inplace, "b = foo(a)",
# And given a {view}_scatter op that shows up later in the graph, "y = {view}_scatter(base, x, args...)"
# Then re-inplacing `foo()` would allow us to remove the `{view}_scatter` op entirely, IF:
# If there are any aliases in the alias_set(a) that satisfy:
# (1) The base of "alias", "alias_base", has the same size/stride/offset metadata as "base"
# (2) The output of running {view}(alias, args...) gives you the same size/stride/offset metadata
#     as "alias"
def _get_view_inverse_node_usages(later_node_usages: Set[Node], self_aliases: Set[Node]) -> Set[Node]:
    def matching_view_metadata(a, b):
        return a.size() == b.size() and \
            a.stride() == b.stride() and \
            a.storage_offset() == b.storage_offset()

    view_inverse_nodes = set()
    # Go through them in node order, so we can see chains of view_scatter ops.
    for n in sorted(later_node_usages, key=lambda x: x.meta['node_idx']):
        if n.target not in _VIEW_INVERSE_MAP:
            continue
        base = n.args[0]
        mutated_view = n.args[1]
        assert isinstance(base, Node)
        assert isinstance(base.meta['fake_result'], FakeTensor)
        assert isinstance(mutated_view, Node)
        assert isinstance(mutated_view.meta['fake_result'], FakeTensor)
        # Check that this view_inverse op actually corresponds to taking doing the inverse
        # of one of our existing self_alias nodes.
        original_view = _VIEW_INVERSE_MAP[n.target]
        for self_alias in self_aliases:
            # We're looking for some alias of the self arg, "alias",
            # that was created from some op `alias = foo(base, args...)`
            # such that the current _scatter op "inverts" that foo call.
            # We can check that by running the original op again, and checking that the strides match.
            if 'view_of' not in self_alias.meta:
                continue
            self_alias_base = self_alias.meta['view_of']
            try:
                # The we're trying to re-use the args from the view_scatter call inside of the corresponding
                # view op, which might throw. This just indicates that view_scatter op isn't a valid inverse
                # of the current alias we're looking at.
                view_replay_metadata = original_view(self_alias_base.meta['fake_result'], *n.args[2:], **n.kwargs)
                expected_metadata = self_alias.meta['fake_result']
                # If the alias and its base both have matching metadata, then this view_scatter op is valid to re-inplace.
                if matching_view_metadata(self_alias_base.meta['fake_result'], base.meta['fake_result']) and \
                        matching_view_metadata(view_replay_metadata, expected_metadata):
                    view_inverse_nodes.add(n)
            except Exception:
                continue

    return view_inverse_nodes


@compatibility(is_backward_compatible=True)
def reinplace(gm, *sample_args):
    """
    Given an fx.GraphModule, modifies it to perform "reinplacing",
    mutating the nodes of the graph.
    We look for out-of-place op call sites like `b = a.add(...)`,
    and convert them to be inplace (`b = a.add_(...)`),
    as long as the input to the current operator ("a") isn't re-used
    anywhere later in the graph.

    This pass currently expects to operate on a **functional, ATen** graph.
    This can be obtained by running `make_fx(functionalize(f))`.

    Sample inputs are needed to determine aliasing relationships of the inputs.
    In general, we can't reinplace node `b = a.add(...)` if "a" aliases any of the
    inputs to the program.

    Given a node "b = foo(a, args...) the algorithm for re-inplacing is as follows:

    (1) Perform some initial checks on the metadata of "a" and "args..."
        that can disqualify them from being reinplaced.

      (1a) Check that the self argument we're attempting to reinplace
           has acceptable dtype/size metadata to reinplace with.

           For example, if we have:
             a = torch.ones(1)
             b = torch.ones(10)
             out = torch.add(a, b)
           We can't turn that into
             a.add_(b)
           Because that would require resizing "a".

           Similarly, we can't convert torch.ge(a, b) into a.ge_(b),
           beause that would require changing a's dtype (from e.g. float32 to bool).
           Note that in this specific example, we could technically do better..

           If we see the pattern:
             a_1 = a.ge(b)
             a_2 = aten._to_copy(a_1, a.dtype)
           Then we this should be valid to completely re-inplace
           (this is exactly what functionalization will emit when it sees a.ge_(b)).

           This optimization is only really important for user programs
           that directly use inplace comparison ops though.

           We also cannot re-inplace on tensors that have overlapping memory,
           e.g. torch.ones(1).expand(4, 4).add_(1)

      (1b) Check if "a" is an alias of any of the program inputs.

          If it is, skip and move to the next node.
          Inplace'ing an op that would cause it to mutate a program is not sound,
          because that would be a side effect visible to the user.

          NOTE: there's a future optimization that we should make:
          if "a" is a (alias of a)  program input, but later in the program
          there is a node that looks like "a.copy_(...)",
          Then re-inplacing is ok to do - we are temporarily re-using a's buffer,
          which will later be overwritten by the copy_() call.

          This will be an important optimization to have for programs that mutate
          their inputs. It currently isn't implemented though.

      (1c) Check if "a" and "args..." alias

          For example, re-inplacing to create code like the below
          isn't guaranteed to be sound:

            aten.mul_(a, a)

    (2) Check that "a" and all of its outstanding aliases are not used anywhere
        later in the graph. If this is the case, then it's safe to re-inplace
        to "b = foo_(a)".

        There are a few caveats to this, explained in more detail below:
        (a) If "a" is used later as an argument to a view op, that is okay.
            It's only a problem if "a" (or that view) is later passed
            into a normal operator, or if it is returned as the program output.
        (b) If "a" is a repeat argument in `foo()`, then don't reinplace.
            Most ATen kernels don't make any guarantees that this is sound,
            e.g. if you do aten.mul_(a, a).
            So we'll just ban re-inplacing in this case.
            It's only a problem if "a" (or that view) is later passed
        (c) If "a" is used as an input into a view "inverse" / "scatter"
            operator, it is potentially fine to re-inplace
            (and remove that scatter operator from the graph).
            See below for a more detailed example.

        NOTE: there is an optimization in this step that is crucial
        to fully recovering performance from functionalization.

        Given this program:
        def f(x):
            a = torch.ops.aten.add(x, x)
            b = torch.ops.aten.diagonal(a)
            torch.ops.aten.fill_(b, 0)
            return d

        Functionalization will emit the following:
        def f(x):
            a = torch.ops.aten.add(x, x)
            b = torch.ops.aten.diagonal(a, 0, 1)
            b_updated = torch.ops.aten.fill(b, 0)
            a_updated = torch.ops.aten.diagonal_scatter(a, b_updated, 0, 1)
            return a_updated

        Ordinarily, we would not be able to reinplace the fill,
        because "b" aliases with "a" which is used by the diagonal_scatter call.

        "re-inplacing" is on the hook for figuring out that it is ok to
        completely, the expensive diagonal_scatter call, if we re-inplace the add().

        So, for every `alias in alias_set(a)`, instead of checking
        that "alias" is not used anywhere later in the graph,
        we check that
            EITHER:
          (a) alias is not used anywhere later in the graph
            OR:
          (b) alias is used exactly once later on in the graph,
              in the following op:

                out = foo_scatter(alias, x, args...)

              where the following must hold:
                (i) "foo_scatter" is the "inverse" operator for foo.
                    This only applies to "foo" ops that are view operators,
                    which view into a subset of the original tensor's memory.
                    In practice, there are ~4 operators where this applies:
                      diagonal -> diagonal_scatter
                      slice -> slice_scatter
                      select -> select_scatter
                      as_strided -> as_strided_scatter
                (ii) "args..." are the same between the foo() and foo_scatter() calls.

    (3) Perform the actual re-inplacing on foo!

      (3b) is the common case, but special care is needed for {view}_scatter (3a)

      (3a) {view}_scatter ops.

        Consider this program:
          a = torch.zeros(2, 2)
          b = torch.ones(2)
          a[0] = b

        Post functionalization, that will look like:
          a = torch.zeros(2)
          b = torch.ones(1)
          a_updated = torch.select_scatter(a, b, 0, 0)

        In this case though, there is no "functional" op to re-inplace!
        Instead, we'd like to directly remove toe select_scatter call.
        We already know from (3) that this is valid,
        because "a" has no later usages in the graph.

        We perform the re-inplacing on the {view}_scatter op like so
        Before:
          a_updated = torch.select_scatter(a, b, args...)
        After:
          a_slice = a.select(a, args...)
          a_slice.copy_(b)

      (3b) Otherwise, replace the functional op with its inplace variant.
        Before:
          b = foo(a, args...)
        After:
          a.foo_(args...)

    (4) Finally, after converting either:
          Before:
            b = foo(a)
          After:
            foo_(a)
        or
          Before:
            b = {slice}_scatter(a, mutated_slice, args...)
          After:
            slice = {slice}(a, args...)
            slice.copy_(mutated_slice)

        We now need to find all later nodes that use "b" as an argument
        and update them to take in "a" instead.

        Note that for the majority of inplace ops, this isn't actually necessary
        (because most inplace ops return "self" as their output).
        This isn't generally true for all mutable ops though, which is why
        we need to actually replace all of the arguments.

        We also need to update our metadata of Dict[StorageWeakRef, Set[Node]],
        That maps a given tensor storage to the set of all nodes that take in that storage
        as an input.
        Specifically, re-inplacing `b = foo(a)` causes "a" and "b"'s sets to get fused
        together.

    (5) Any "view_inverse/scatter" nodes that were identified as "it's ok to ignore them"
        during step (3) get manually deleted from the graph.
        Their outputs are no longer used, so technically standard DCE would be able
        to do this, but we can no longer run FX's DCE pass now that we have mutable
        ops in the graph.
    """
    _FunctionalizationMetadataProp(gm).propagate(*sample_args)

    # Useful debug printing
    # def _print(x):
    # if isinstance(x, FakeTensor):
    # print(f'fake_result: {StorageWeakRef(x._typed_storage()).cdata}')

    # for n in gm.graph.nodes:
    # print(n.format_node())
    # if hasattr(n, 'meta'):
    # print(f'node_idx: {n.meta["node_idx"]}')
    # if 'fake_result' in n.meta:
    # tree_map(_print, n.meta['fake_result'])
    # if 'view_of' in n.meta:
    # print(f'view_of: {str(n.meta["view_of"])}')
    # print()

    # We need to know which nodes correspond to inputs (or their aliases)
    # so we know not to re-inplace them.
    # NOTE: later, we'll need to add an optimization for fully recovering performance
    # on programs that mutate inputs.
    input_storages = set(
        StorageWeakRef(
            node.meta['fake_result']._typed_storage()
        ) for node in gm.graph.nodes if node.op == 'placeholder')


    # We also need to know for a given node, what are all of its aliasing nodes.
    storage_to_nodes: Dict[StorageWeakRef, Set[Node]] = defaultdict(set)
    for n in gm.graph.nodes:
        if 'fake_result' in n.meta:
            # Tree-mapping because some ops can return lists of tensors.
            def _add_to_map(x):
                if isinstance(x, FakeTensor):
                    storage_to_nodes[StorageWeakRef(x._typed_storage())].add(n)
            tree_map(_add_to_map, n.meta['fake_result'])

    # inplace-ify functional ops, subject to the constraints written below.
    all_later_view_inverse_nodes_to_delete = set()
    for idx, node in enumerate(gm.graph.nodes):
        if node.op == 'call_function':

            # Today, the re-inplace pass on directly acts on:
            # - functional ops with an inplace variant
            # - {view}_scatter ops that can be potentially removed from the graph.
            # Both of these ops take in tensor first args, so filtering on this condition
            # makes the later code simpler.
            # We should revisit this at some point though, particularly when we also want
            # the reinplacer to be able to handle out= and mutable operators
            # and tensorlist first args (like `_foreach_` ops).
            if not isinstance(node.target, torch._ops.OpOverload):
                continue
            if len(node.target._schema.arguments) < 1:
                continue
            if type(node.target._schema.arguments[0].type) != torch.TensorType:
                continue

            # Step 1a: Check that the self argument we're attempting to reinplace
            # has the same size/stride as the output.
            # For example, we shouldn't try to reinplace torch.add(scalar_tensor, larger_tensor)
            # As it would require resizing scalar_tensor.
            # (We could potentially swizzle this into larger_tensor.add_(scalar_tensor),
            # this is probably an optimization to revisit later).
            self_arg = node.args[0]
            self_flattened, _ = tree_flatten(self_arg.meta['fake_result'])
            node_flattened, _ = tree_flatten(node.meta['fake_result'])
            self_has_wrong_metadata = False
            if len(self_flattened) == len(node_flattened):
                for self_meta, node_meta in zip(self_flattened, node_flattened):
                    if self_meta.numel() != node_meta.numel():
                        self_has_wrong_metadata = True
                    if self_meta.dtype != node_meta.dtype:
                        self_has_wrong_metadata = True
                    # We also cannot re-inplace on tensors that have internal memory overlap.
                    # e.g. torch.ones(1).expand(4, 4).add_(1)
                    if torch._debug_has_internal_overlap(self_meta) == 1:
                        self_has_wrong_metadata = True
            # Here, we (optimistically) assume that a.resize(b) is valid to re-inplace,
            # Since users should never really be calling the functional "torch.ops.aten.resize"
            # op directly in their programs.
            if self_has_wrong_metadata and node.target != torch.ops.aten.resize.default:
                continue

            # Step 1b: ensure that the op we're trying to re-inplace isn't a program input
            self_arg_name = self_arg.name
            self_arg_storage = StorageWeakRef(self_arg.meta['fake_result']._typed_storage())
            if self_arg_storage in input_storages:
                # TODO: later, add the optimization for handling `copy_()` calls in the graph.
                continue
            if len([x for x in node.args if x is self_arg]) > 1:
                # Step 1c:
                # Calling stuff like aten.mul_(a, a) isn't guaranteed to be sound,
                # so we prevent re-inplacing in this case.
                continue

            self_arg_storage = StorageWeakRef(self_arg.meta['fake_result']._typed_storage())
            self_aliases = storage_to_nodes[self_arg_storage]

            # First, we find all later usages of any of the aliases of self_arg.
            later_node_usages = _get_all_later_node_usages(self_aliases, node.meta['node_idx'])
            # Then, we check if any of those later usages are actually view_scatter ops
            # that are safe to fully remove.
            later_view_inverse_node_usages = _get_view_inverse_node_usages(later_node_usages, self_aliases)

            # Step 2: Check to see if the input to the op is re-used later in the graph.
            # If not (same goes for its aliases), then this op is safe to re-in place.
            # This is a slightly roundabout way to check that there are no later usages of the current self argument.
            # (later_view_inverse_node_usages corresponds to "view_scatter" nodes that we are allowed to delete)
            can_reinplace = len(later_node_usages - later_view_inverse_node_usages) == 0
            if not can_reinplace:
                continue

            # Step 3a: Special handling for when we see *_scatter operators.
            # When we see an operator like `b = torch.slice_scatter(a, ...)`,
            # instead of trying to "inplace" it into a.slice_scatter_(..._),
            # we would prefer to remove it from the graph entirely,
            # and instead copy_() the slice directly into the larger tensor.
            # See the description of the algorithm for a full example.
            if node.target in _VIEW_INVERSE_MAP and node not in all_later_view_inverse_nodes_to_delete:
                view_op = _VIEW_INVERSE_MAP[node.target]
                # Before:
                #   base_updated = torch.ops.aten.slice_scatter.default(base, mutated_slice, args...)
                # After:
                #   slice = torch.ops.aten.slice.default(base, args...)
                #   slice.copy_(mutated_slice)
                with gm.graph.inserting_before(node):
                    mutated_slice_node = node.args[1]
                    remaining_slice_args = node.args[2:]
                    slice_node = gm.graph.create_node(
                        'call_function', view_op, (self_arg,) + tuple(remaining_slice_args), node.kwargs)
                    copy_node = gm.graph.create_node(
                        'call_function', torch.ops.aten.copy_.default, (slice_node, mutated_slice_node,), {})
                # Add the slice_scatter node to our "nodes to delete" list.
                all_later_view_inverse_nodes_to_delete.add(node)


            else:
                # Step 3b: Check to see if this operator has an inplace variant.
                maybe_inplace_op = _maybe_get_inplace_op(node.target)
                if maybe_inplace_op is None:
                    continue
                # And if so, replace it with its inplace variant.
                node.target = maybe_inplace_op

            # At this point, 'storage_to_nodes' will be stale.
            # Now that we're inplacing `b = foo(a)`, we need to effectively
            # union together the dict values for b and a's storage.
            # Hmm... morally I think we also want to keep the `fake_result` metadata
            # up to date here, but I'm not sure how easy it is to do.
            # Maybe it's fine to wait until the end of the pass to update it.
            curr_node_storage = StorageWeakRef(node.meta['fake_result']._typed_storage())
            storage_to_nodes[self_arg_storage].update(storage_to_nodes[curr_node_storage])
            storage_to_nodes[curr_node_storage].update(storage_to_nodes[self_arg_storage])

            # Need to remember the view_scatter view nodes we found so we can remove them alter.
            all_later_view_inverse_nodes_to_delete.update(later_view_inverse_node_usages)

            # Step 4:
            # Now that we've replaced b = a.foo() with a.foo_(),
            # We need to replace any later usages of "b" with "a"
            for old in itertools.chain([node], later_view_inverse_node_usages):
                new = old.args[0]
                nodes_to_update = [n for n in old.users if n.meta['node_idx'] > node.meta['node_idx']]
                for node_to_update in nodes_to_update:
                    new_args = []
                    args = node_to_update.args

                    def replace_arg(a):
                        if a == old:
                            return new
                        return a

                    # First, replace usages of "b" with "a"
                    node_to_update.args = tree_map_only(Node, replace_arg, node_to_update.args)
                    node_to_update.kwargs = tree_map_only(Node, replace_arg, node_to_update.kwargs)

                    # Second, update our storage_to_nodes data structure.
                    old_flattened_res, _ = tree_flatten(old.meta['fake_result'])
                    node_flattened_res, _ = tree_flatten(node_to_update.meta['fake_result'])

                    old_res_storage = set(
                        StorageWeakRef(
                            x._typed_storage()
                        ) for x in old_flattened_res if isinstance(x, FakeTensor))
                    node_res_storage = set(
                        StorageWeakRef(
                            x._typed_storage()
                        ) for x in node_flattened_res if isinstance(x, FakeTensor))

                    # This will happen if we're updating a view op, e.g.
                    # e.g. replacing
                    #     x = view(old)
                    #     x = view(new)
                    # When that happens, we need to make sure to keep our
                    # storage mapping up to date.
                    #
                    # We're checking for len(...) == 1 here because all view ops are guaranteed to return either a single tensor,
                    # or multiple tensors that all share the same storage.
                    # We can't just check equality because we might encounter FX nodes that return zero tensor outputs.
                    if len(old_res_storage) == 1 and len(node_res_storage) == 1 and old_res_storage == node_res_storage:
                        new_flattened_res, _ = tree_flatten(new.meta['fake_result'])
                        new_res_storage = set(
                            StorageWeakRef(
                                x._typed_storage()
                            ) for x in new_flattened_res if isinstance(x, FakeTensor))
                        assert len(new_res_storage) == 1
                        (old_ref,) = old_res_storage
                        (new_ref,) = new_res_storage
                        (node_ref,) = node_res_storage
                        # Technically, "old_ref" and all its aliases will remain
                        # in our mapping.
                        # That should be fine though, since we deleted "old"
                        # from the graph at this point.
                        storage_to_nodes[node_ref].update(storage_to_nodes[new_ref])
                        storage_to_nodes[new_ref].update(storage_to_nodes[node_ref])

    # Step 4: delete any _scatter nodes that we de-functionalized
    # Need to take care not to delete any of these nodes until after *all* modifications
    # to the graph are finished.
    for to_delete in all_later_view_inverse_nodes_to_delete:
        gm.graph.erase_node(to_delete)


    gm.recompile()
    return gm
