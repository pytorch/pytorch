import torch.fx
from torch.fx import Node
from torch.fx._compatibility import compatibility
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.utils._pytree import tree_map
from torch.multiprocessing.reductions import StorageWeakRef

import _operator
from enum import Enum
import itertools
from typing import Optional, Set
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
# n.meta['fake_result']: FakeTensor
#   The fake tensor output from running the current node
# n.meta['node_usages']: List[List[int, Node]]
#   A list of all nodes that take in the current node as an input
#   Each element is actually an (int, Node) pair, specifying the index of the node in the graph.
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
        node.meta['node_usages'] = []

        # (1) Update metadata with the list of nodes that are used by this node
        # copy_() doesn't read from its first argument; it writes to it, overwriting previous data.
        # We don't want to treat it as "being used as an input".
        node_args = node.args
        if node.target is torch.ops.aten.copy_.default:
            node_args = node_args[1:]

        def _update_node_usage(x):
            if isinstance(x, torch.fx.node.Node):
                # We map to the actual fx.Node object, because later we might need to
                # modify the arguments that this node is called with
                x.meta['node_usages'].append(node)

        tree_map(_update_node_usage, list(itertools.chain(node_args, node.kwargs.values())))

        # (2) Update metadata to track aliasing information about view tensor nodes.
        if node.op == 'call_function':
            view_type = _get_view_type(node.target)
            if view_type == _ViewType.SingleOutputView:
                assert isinstance(node.args[0], torch.fx.node.Node)
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
            #    getitem.meta['view_of'] = x_1
            #    getitem1.meta['view_of'] = x_1
            if node.target is _operator.getitem:
                list_arg = node.args[0]
                maybe_base_of_view = self.multi_output_view_nodes.get(list_arg, None)
                if maybe_base_of_view is not None:
                    # Note: we could also track indexing info here for multi-output views.
                    # I don't think this metadata is strictly needed for de-functionalization.
                    assert isinstance(maybe_base_of_view, torch.fx.node.Node)
                    node.meta['view_of'] = maybe_base_of_view
        return result



    def propagate(self, *args):
        self.multi_output_view_nodes = {}
        self.node_counter = -1
        with FakeTensorMode.push() as mode:
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
    # This is for sanity: if foo() and foo_() are both operators,
    # we expect them to have compatible schemas.
    # (This is asserted by codegen for ATen, but might not be true
    # for other arbitrary operators).
    assert len(inplace_overloads_with_matching_schemas) == 1
    inplace_op = inplace_overloads_with_matching_schemas[0]
    return inplace_op

def _maybe_get_base_of_curr_node(self_arg):
    # If the current arg was created from a view op, return the node
    # corresponding to tensor that it was viewed from.
    if _is_view_op(self_arg.target):
        assert isinstance(node.args[0], torch.fx.node.Node)
        return self_arg.args[0]
    # We might need to look one level down:
    # for multi-output view ops like .split(),
    # The split op creates a list of tensors, and _operator.getitem
    # is recorded into the FX graph to pass individual tensors from that list
    # into other operators.
    if node.target is _operator.getitem:
        list_arg = self_arg.args[0]
        if _is_view_op(list_arg.target):
            assert isinstance(list_arg.args[0], torch.fx.node.Node)
            return list_arg.args[0]
    return None

def _can_reuse_tensor_memory(self_arg, *, view_of_self: Optional[torch.fx.node.Node] = None):
    nodes_used = self_arg.meta['node_usages'] if 'node_usages' in self_arg.meta else []
    # If the current tensor argument is used as an input to any ops later in the graph,
    # then we can't re-use its memory
    if any(_ for n_idx, n in nodes_used if n_idx > idx):
        return False
    # Special case; if the current node is base, and we're trying to re-inplace
    # an operator to v where v = view_op(base, args...),
    # Then it is safe to re-inplace v, which will directly modify, base,
    #     **as long as** either:
    # (1) base is also not used anywhere later in the program
    # (2) base used exactly once later in the program, in the following operation:
    #     base_updated = view_inverse_op(base, v_updated, args...)
    # where "args..." must match up exactly, and "view_inverse_op"
    # must be the "inverse" of view, according to the inverse rules
    # in the functionalization pass.
    maybe_base_of_self = _maybe_get_base_of_curr_node(self_arg)
    if maybe_base_of_self is not None:
        return _can_reuse_tensor_memory(base_of_self, view_of_self=self_arg)
    return True

_VIEW_INVERSE_MAP = {
    torch.ops.aten.diagonal.default: torch.ops.aten.diagonal_scatter.default,
    torch.ops.aten.select.int: torch.ops.aten.select_scatter.default,
    torch.ops.aten.slice.Tensor: torch.ops.aten.slice_scatter.default,
    torch.ops.aten.as_strided.default: torch.ops.aten.as_strided_scatter.default,
}

# Uses the 'view_of' metadata set by FunctionalizationMetadataProp
# to determine if two different nodes are identical views of the same base.
# For example:
#    a1 = x.view(2, 4)
#    a2 = a1[0]
#    b1 = x.view(2, 4)
#    b2 = b1[0]
# Here, b2 and a2 are matching view tensors, because their chain of views is identical.
def _are_matching_view_tensors(view1: torch.fx.node.Node, view2: torch.fx.node.Node) -> bool:
    base1 = view1.meta.get('view_of', None)
    base2 = view2.meta.get('view_of', None)
    if base1 is None and base2 is None:
        return True
    if base1 is None or base2 is None:
        return False
    # Check that "args..." are the same too.
    assert len(view1.args) == len(view2.args)
    assert len(view1.kwargs) == len(view2.kwargs)
    # If the two views were created through the same view op (with the same args),
    # then recursively check if their bases are also matching views.
    if view1.target is view2.target:
        if all(x == y for (x, y) in zip(
               itertools.chain(view2.args[1:], view2.kwargs),
               itertools.chain(view1.args[1:], view1.kwargs))):
            return _are_matching_view_tensors(base1, base2)
    return False



def _satisfies_inverse_view(view_inverse_node: torch.fx.node.Node, tensor_aliases: Set[torch.fx.node.Node]):
    for alias in tensor_aliases:
        # If the current node is a *_scatter op, first check if any of the aliases
        # correspond to the inverse (e.g. slice_scatter -> slice)
        if view_inverse_node.target == _VIEW_INVERSE_MAP.get(alias.target, None):
            # Every view_inverse op is of the form *_scatter(base, mutated_view, args...)
            # First, check that "base" in the inverse call matches the base in the view op call.

            if _are_matching_view_tensors(view_inverse_node.args[0], alias.args[0]):
                # Then, check that "args..." are the same too.
                # view inverse nodes like *_scatter are always of the form:
                #     select(base, args...)
                #     select_scatter(base, mutated_view, args...)
                assert len(alias.args) + 1 == len(view_inverse_node.args)
                assert len(alias.kwargs) == len(view_inverse_node.kwargs)
                if all(x == y for (x, y) in zip(
                       itertools.chain(view_inverse_node.args[2:], view_inverse_node.kwargs),
                       itertools.chain(alias.args[1:], alias.kwargs))):
                    return True
    return False


# This function tests where a tensor's memory is allowed to be re-inplaced.
# e.g. given:
#    a.add(b)
# It returns true if we can convert it into
#    a.add_(b)
# We figure this out as follows:
#   For every alias of a, "x",
#   EITHER
#       "x" is *never* used later in the program
#   OR
#       there is some existing alias of a, "alias", where
#       "alias" is used in some slicing view operation like _ = torch.select(alias, args...)
#       AND "x" is used in the corresponding "scatter" variant of that operator,
#       e.g. _ = torch.select_scatter(x, mutated_view, args...)
#       AND the arguments to the view and its scatter operator are the same (args... is equal).
#   This is effectively undoing the view-inverse logic performed by functionalization.
def _get_all_later_node_usages(curr_node: torch.fx.node.Node, tensor_aliases: Set[torch.fx.node.Node], op_index: int, *, only_include_view_inverse_nodes: bool = False) -> Set[torch.fx.node.Node]:
    def _add_if_tensor(x, set_):
        if isinstance(x, torch.Tensor):
            set_.add(StorageWeakRef(x.storage()))

    # Get all storages of the current node
    curr_node_tensor_storages = set()
    tree_map(lambda x: _add_if_tensor(x, curr_node_tensor_storages), curr_node.meta['fake_result'])

    # For every alias of the current node, check that either:
    # (1) The alias isn't used anywhere later in the graph.
    # (2) Or that it's used exactly once, in a "view_inverse" (scatter) operation.
    nodes_used_after = set()
    for t in tensor_aliases:
        # get all nodes that use the current alias
        usage_nodes = t.meta['node_usages']
        for n in usage_nodes:
            # We only care about usages after the current node
            if n.meta['node_idx'] <= op_index:
                continue
            # And we don't care about view usages (they only matter if that view is then later used in another node)
            # We really just need to check if this node shares storage with curr_node,
            # but we need to take care to handle the more general case: if curr_node and n are both lists of tensors,
            # then their storages should fully intersect.
            n_tensor_storages = set()
            tree_map(lambda x: _add_if_tensor(x, n_tensor_storages), n.meta['fake_result'])

            # What's the idea here? if "n" is a non-view node, like add_tensor = torch.add(x, ...),
            # Then the FakeTensor storage of the add_tensor node will be different
            # from all of the tensor all of the current tensor's storages.
            # Also, if n isn't an operator, then its fake result won't be a tensor.
            if len(n_tensor_storages) > 0:
                if curr_node_tensor_storages & n_tensor_storages == n_tensor_storages:
                    continue
            if only_include_view_inverse_nodes and not _satisfies_inverse_view(n, tensor_aliases):
                continue
            nodes_used_after.add(n)
    return nodes_used_after



@compatibility(is_backward_compatible=True)
def reinplace(gm, *sample_args):
    """
    Given an fx.GraphModule, modifies it to perform "reinplacing",
    mutating the nodes of the graph.
    We look for out-of-place op call sites like `b = a.add(...)`,
    and convert them to be inplace (`b = a.add_(...)`),
    as long as the input to the current operator ("a") isn't re-used
    anywhere later in the graph.

    Sample inputs are needed to determine aliasing relationships of the inputs.
    In general, we can't reinplace node `b = a.add(...)` if "a" aliases any of the
    inputs to the program.

    There is one exception though: if "a" is copied to at any point later in the program,
    e.g. `a.copy_(...)`, then we are free to re-use a's buffer any time
    before that node executes.

    This is an important optimization to include, to ensure that after running
    functionalization and then reinplacing, we don't regress the net memory usage
    of the original model.
    """
    _FunctionalizationMetadataProp(gm).propagate(*sample_args)

    def _print(x):
        if isinstance(x, torch.Tensor):
            print(f'fake_result: {StorageWeakRef(x.storage()).cdata}')

    for n in gm.graph.nodes:
        print(n.format_node())
        if hasattr(n, 'meta'):
            print(f'node_idx: {n.meta["node_idx"]}')
            if 'fake_result' in n.meta:
                tree_map(_print, n.meta['fake_result'])
            if 'node_usages' in n.meta:
                print(f'node_usages: {", ".join([str(x) for x in n.meta["node_usages"]])}')
            if 'view_of' in n.meta:
                print(f'view_of: {str(n.meta["view_of"])}')
        print()

    # We need to know which nodes correspond to inputs (or their aliases)
    # so we know not to re-inplace them.
    # NOTE: later, we'll need to add an optimization for fully recovering performance
    # on programs that mutate inputs.
    input_storages = set(StorageWeakRef(node.meta['fake_result'].storage()) for node in gm.graph.nodes if node.op == 'placeholder')


    # We also need to know for a given node, what are all of its aliasing nodes.
    storage_to_nodes: Dict[StorageWeakRef, Set[torch.fx.node.Node]] = defaultdict(set)
    for n in gm.graph.nodes:
        if 'fake_result' in n.meta:
            # Tree-mapping because some ops can return lists of tensors.
            tree_map(lambda x: storage_to_nodes[StorageWeakRef(x.storage())].add(n), n.meta['fake_result'])


    # inplace-ify functional ops, subject to the constraints written below.
    for idx, node in enumerate(gm.graph.nodes):
        if node.op == 'call_function':
            # Step 1: Check to see if this operator has an inplace variant.
            maybe_inplace_op = _maybe_get_inplace_op(node.target)
            if maybe_inplace_op is not None:
                # This is a proxy check for ensuring that the first argument is "tensor-like"
                # (This should be the case for all ops with inplace variants in ATen,
                # although we technically don't have guarantees for custom ops).
                assert len(node.target._schema.arguments) > 0
                assert 'Tensor' in str(node.target._schema.arguments[0].type)

                # Step 2: ensure that the op we're trying to re-inplace isn't a program input.
                self_arg = node.args[0]
                self_arg_name = self_arg.name
                self_arg_storage = StorageWeakRef(self_arg.meta['fake_result'].storage())
                if self_arg_storage in input_storages:
                    # TODO: later, add the optimization for handling `copy_()` calls in the graph.
                    continue

                # Step 3: Check to see if the input to the op is re-used later in the graph.
                # If not (same goes for its aliases), then this op is safe to re-in place.
                self_aliases = storage_to_nodes[StorageWeakRef(self_arg.meta['fake_result'].storage())]
                later_view_inverse_node_usages = _get_all_later_node_usages(self_arg, self_aliases, node.meta['node_idx'], only_include_view_inverse_nodes=True)
                later_node_usages = _get_all_later_node_usages(self_arg, self_aliases, node.meta['node_idx'])

                if len(later_node_usages - later_view_inverse_node_usages) == 0:
                    # Step 4: replace the current out-of-place op with its inplace variant.
                    node.target = maybe_inplace_op

                    # Now that we've replaced b = a.foo() with a.foo_(),
                    # We need to replace any later usages of "b" with "a"
                    for old in itertools.chain([node], later_view_inverse_node_usages):
                        new = old.args[0]
                        nodes_to_update = [n for n in old.meta['node_usages'] if n.meta['node_idx'] > node.meta['node_idx']]
                        for node_to_update in nodes_to_update:
                            new_args = []
                            for arg_idx, a in enumerate(node_to_update.args):
                                if isinstance(a, torch.fx.node.Node) and a.name == old.name:
                                    new_args.append(new)
                                else:
                                    new_args.append(a)
                            new_kwargs = {}
                            for kwarg_idx, (k, v) in enumerate(node_to_update.kwargs):
                                if isinstance(v, torch.fx.node.Node) and v.name == old.name:
                                    new_kwargs[k] = new
                                else:
                                    new_kwargs[k] = v
                            node_to_update.args = new_args
                            node_to_update.kwargs = new_kwargs

                    # Step 5: delete any _scatter nodes that we de-functionalized
                    for to_delete in later_view_inverse_node_usages:
                        gm.graph.erase_node(to_delete)


    gm.recompile()
    return gm
