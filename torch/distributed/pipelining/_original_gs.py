import collections
import weakref

import torch
import torch.nn as nn
from torch.autograd.graph import GradientEdge


def _get_grad_fn_or_grad_acc(t):
    if t.requires_grad and t.grad_fn is None:
        return t.view_as(t).grad_fn.next_functions[0][0]
    else:
        return t.grad_fn


def reverse_closure(roots, target_nodes):
    # Recurse until we reach a target node
    closure = set()
    actual_target_nodes = set()
    q: Deque = collections.deque()
    for node in roots:
        if node is not None and node not in closure:
            closure.add(node)
            q.append(node)
    while q:
        node = q.popleft()
        reverse_edges = node.metadata.get("reverse_edges", [])
        for holder_ref, idx in reverse_edges:
            ref = holder_ref()
            if ref is None:
                raise RuntimeError("Reverse graph is no longer alive")
            fn = ref.node
            if fn in closure or fn is None:
                continue
            if fn in target_nodes:
                actual_target_nodes.add(fn)
                continue
            closure.add(fn)
            q.append(fn)
    return closure, actual_target_nodes


# Enable weak pointer
class Holder:
    def __init__(self, node):
        self.node = node


# TODO: use weak references to avoid reference cycle
def construct_reverse_graph(roots):
    q: Deque = collections.deque()
    root_seen = set()
    reverse_graph_refs = []
    for node in roots:
        if node is not None and node not in root_seen:
            q.append(node)
            root_seen.add(node)
    while q:
        node = q.popleft()
        for fn, idx in node.next_functions:
            if fn is not None:
                # Don't necessarily need to store on the graph
                reverse_edges = fn.metadata.get("reverse_edges", [])
                if len(reverse_edges) == 0:
                    q.append(fn)
                holder = Holder(node)
                holder_ref = weakref.ref(holder)
                reverse_graph_refs.append(holder)
                reverse_edges.append((holder_ref, idx))
                fn.metadata["reverse_edges"] = reverse_edges
    return reverse_graph_refs


def get_param_groups(inputs, params):
    inputs_closure, _ = reverse_closure(inputs, set())
    param_groups = dict()  # keyed on intermediates
    for i, param in enumerate(params):
        closure, intersected = reverse_closure([param], inputs_closure)
        param_group = {
            "params": set([param]),
            "intermediates": set(intersected),
        }
        for input_node in intersected:
            existing = param_groups.get(input_node, None)
            if existing is not None:
                existing["params"] = existing["params"].union(param_group["params"])
                existing["intermediates"] = existing["intermediates"].union(
                    param_group["intermediates"]
                )
                param_group = existing
            else:
                param_groups[input_node] = param_group

    # Sanity check: union of all param_groups params should be equal to all params
    union_params = set()
    seen_ids = set()
    unique_param_groups = []
    for param_group in param_groups.values():
        if id(param_group) not in seen_ids:
            seen_ids.add(id(param_group))
            unique_param_groups.append(param_group)
            union_params = union_params.union(param_group["params"])
    assert union_params == set(params)

    return unique_param_groups


def compute_grads_only_inputs2(roots, inps, weights):
    root_grad_fns = list(map(_get_grad_fn_or_grad_acc, roots))
    inp_grad_fns = list(map(_get_grad_fn_or_grad_acc, inps))
    weight_grad_fns = list(map(_get_grad_fn_or_grad_acc, weights))

    reverse_graph_refs = construct_reverse_graph(root_grad_fns)
    param_groups = get_param_groups(inp_grad_fns, weight_grad_fns)
    del reverse_graph_refs

    for param_group in param_groups:
        for i, intermediate in enumerate(param_group["intermediates"]):

            def get_hook(param_group, i):
                def hook(grad_inputs):
                    if param_group.get("grads", None) is None:
                        param_group["grads"] = [None] * len(
                            param_group["intermediates"]
                        )
                    param_group["grads"][i] = grad_inputs

                return hook

            # These are always "split" nodes that we need to recompute, so
            # save their inputs.
            intermediate.register_prehook(get_hook(param_group, i))

    dinputs = torch.autograd.grad(
        (out,),
        inputs=tuple(inps),
        grad_outputs=(torch.ones_like(out),),
        retain_graph=True,
    )
    return dinputs, param_groups


def compute_grads_only_weights2(user_weights, param_groups):
    all_dweights = dict()
    for param_group in param_groups:
        # TODO: Handle case where intermediate can have multiple outputs
        intermediate_edges = tuple(
            GradientEdge(i, 0) for i in param_group["intermediates"]
        )
        weights_edges = tuple(GradientEdge(w, 0) for w in param_group["params"])

        assert all(len(g) == 1 for g in param_group["grads"])
        # [NEW!] Able to pass a GradientEdge to autograd.grad as output
        # We do not need to retain_graph because... guarantee no overlap?
        print("trying to execute: ", intermediate_edges, weights_edges)
        dweights = torch.autograd.grad(
            intermediate_edges,
            weights_edges,
            grad_outputs=sum(param_group["grads"], tuple()),
        )
        for w, dw in zip(param_group["params"], dweights):
            all_dweights[w] = dw
    # return grads in the original order weights were provided in
    out = []
    for w in user_weights:
        grad_acc = _get_grad_fn_or_grad_acc(w)
        out.append(all_dweights[grad_acc])
    return tuple(out)


# Setup
mod1 = nn.Linear(10, 10)
mod2 = nn.Linear(10, 10)

a = torch.rand(10, requires_grad=True)

weights = tuple(mod1.parameters()) + tuple(mod2.parameters())
inps = (a,)

out = mod2(mod1(a))


class LoggingTensorMode(torch.utils._python_dispatch.TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        rs = func(*args, **kwargs)
        print(f"{func.__module__}.{func.__name__}")
        return rs


print(" -- SPLIT -- ")
# Compute gradients in two parts
with LoggingTensorMode():
    print("PART 1")
    dinputs, state = compute_grads_only_inputs2((out,), inps, weights)
    print("PART 2")
    dweights = compute_grads_only_weights2(weights, state)

out = mod2(mod1(a))

print(" -- REF -- ")

# Compare with reference
with LoggingTensorMode():
    ref_all_gradients = torch.autograd.grad(
        out, inputs=tuple(inps) + weights, grad_outputs=(torch.ones_like(out),)
    )

for actual, ref in zip(dinputs + dweights, ref_all_gradients):
    print(torch.allclose(actual, ref))
