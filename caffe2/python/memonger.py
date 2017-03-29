## @package memonger
# Module caffe2.python.memonger
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import networkx as nx
import collections
import time
import copy
from caffe2.python import workspace

import logging

log = logging.getLogger("memonger")
log.setLevel(logging.INFO)
LiveRange = collections.namedtuple('LiveRange', ["defined", "used"])


def share_grad_blobs(net, losses, param_grads, namescope):
    '''
    Implements similar optimization as Torch's shareGradInput():
    for the gradients that are passed between layers, share blobs between
    operators when possible. This yields significant memory savings with
    deep networks.

    Returns an optimized protobuf (assign to net._net)
    '''
    def is_grad_blob(b):
        name = str(b)
        # Note: need to look at _{namescope} pattern as it matches
        # to handle the auto-split gradients
        return "_grad" in name and (name.startswith(namescope) or
            name.startswith("_" + namescope)) and name not in param_grads

    def is_grad_op(op):
        # TODO: something smarter
        for inp in op.input:
            if is_grad_blob(inp):
                return True
        for out in op.output:
            if is_grad_blob(out):
                return True
        return False

    start_time = time.time()
    log.warn("NOTE: Executing *experimental* memonger to " +
             "optimize gradient memory")

    # Collect ops that have something to do with
    # gradients
    if not namescope.endswith("/"):
        namescope += "/"

    netproto = copy.deepcopy(net.Proto())
    grad_ops = [op for op in netproto.op if is_grad_op(op)]

    # Create mapping from blobs to ops
    blobs_to_ops = collections.defaultdict(lambda: [])
    blob_input_count = collections.defaultdict(lambda: 0)
    op_inputs = collections.defaultdict(lambda: 0)
    op_visit_count = collections.defaultdict(lambda: 0)
    for i, op in enumerate(grad_ops):
        for inp in op.input:
            if is_grad_blob(inp) or inp in losses:
                # Ignore in-place transformation ops (self cycles)
                if inp not in op.output:
                    blobs_to_ops[inp].append(i)
                    op_inputs[i] += 1

    # Traverse operators starting from the loss blobs.
    # Keep tabs on when blobs are seen first and last, and also
    # when operators have their input satisfied. Share blobs only
    # under same branch, avoiding problems with parallel workers.
    output_blobs = set()
    mapping = {}

    def descend(op_idx, free_blobs):
        cur_op = grad_ops[op_idx]
        new_free_blobs = set()
        for inp in cur_op.input:
            if is_grad_blob(inp):
                blob_input_count[inp] += 1
                if blob_input_count[inp] == len(blobs_to_ops[inp]):
                    actual_blob = inp if inp not in mapping else mapping[inp]
                    new_free_blobs.add(actual_blob)

        for outp in cur_op.output:
            if is_grad_blob(outp):
                if outp not in output_blobs:
                    # First seen this blob as output, can assign to a free blob
                    for freeb in free_blobs:
                        mapping[outp] = freeb
                        free_blobs.remove(freeb)
                        break

                output_blobs.add(outp)

        free_blobs.update(new_free_blobs)

        first_branch = True
        for outp in cur_op.output:
            for inp_op_idx in blobs_to_ops[outp]:
                op_visit_count[inp_op_idx] += 1

                # Descend only if we have satisfied all inputs
                if op_visit_count[inp_op_idx] == op_inputs[inp_op_idx]:
                    free_blobs_fwd = free_blobs if first_branch else set()
                    first_branch = False
                    descend(inp_op_idx, free_blobs_fwd)

    # Start DFS from the losses
    for loss in losses:
        for op_idx in blobs_to_ops[loss]:
            descend(op_idx, set())

    # Rename the shared blobs
    shared_blobs = set(mapping.values())
    renamed = {}
    for j, b in enumerate(shared_blobs):
        renamed[b] = namescope + "__m{}_".format(j)

    # Final mapping
    for k, v in mapping.items():
        mapping[k] = renamed[v]

    # Add the originators
    mapping.update(renamed)
    log.info("Remapping {} blobs, using {} shared".format(
        len(mapping), len(renamed),
    ))
    apply_assignments(netproto, mapping)
    log.info("Gradient memory optimization took {} secs".format(
        time.time() - start_time),
    )
    return netproto


def topological_sort_traversal(g):
    return nx.topological_sort(g)


def compute_ranges(linearized_ops):
    blobs = collections.defaultdict(lambda: LiveRange(defined=None, used=None))
    for i, op in enumerate(linearized_ops):
        for blob in op.input:
            used = blobs[blob].used
            if used is None:
                used = i
            else:
                used = max(used, i)
            blobs[blob] = blobs[blob]._replace(used=used)
        for blob in op.output:
            defined = blobs[blob].defined
            if defined is None:
                defined = i
            else:
                defined = min(defined, i)
            blobs[blob] = blobs[blob]._replace(defined=defined)

    return blobs


def is_compatible(candidate_range, assignment, static_blobs):
    (name, range_) = assignment[-1]
    if name in static_blobs:
        return False
    if candidate_range.defined is None or range_.defined is None \
      or range_.used is None:
        return False
    return candidate_range.defined > range_.used


def compute_blob_assignments(assignments):
    blob_assignments = {}
    for assignment in assignments:
        if len(assignment) == 1:
            continue
        last_blob, _ = assignment[-1]
        for (blob, _) in assignment:
            blob_assignments[blob] = last_blob
    return blob_assignments


def compute_assignments(ranges, static_blobs):
    # Sort the ranges based on when they are last used.
    # If LiveRange.used is None, then the blob is never used and could
    # be consumed externally. Sort these to the end of the list as opposed
    # to the beginning so that they can be shared as well.
    ranges = sorted(
        list(ranges.items()),
        key=lambda p: (p[1].used is None, p[1].used),
    )
    assignments = []
    for (name, range_) in ranges:
        assigned = False
        for assignment in assignments:
            if is_compatible(range_, assignment, static_blobs):
                assignment.append((name, range_))
                assigned = True
                break
        if assigned:
            continue
        assignments.append([(name, range_)])
    return assignments


def compute_interference_graph(ops):
    g = nx.DiGraph()
    for i, op in enumerate(ops):
        g.add_node(i, op=op)
    for i, parent_op in enumerate(ops):
        for j, child_op in enumerate(ops):
            if i == j:
                continue
            if any(output in child_op.input for output in parent_op.output):
                deps = set(child_op.input).intersection(parent_op.output)
                g.add_edge(i, j, deps=deps)
                assert nx.is_directed_acyclic_graph(g), child_op
    return g


Optimization = collections.namedtuple(
    'Optimization', ['net', 'assignments', 'blob_assignments'])


def apply_assignments(net, blob_assignments):
    def canonical_name(blob):
        if blob not in blob_assignments:
            return blob
        return blob_assignments[blob]

    for op in net.op:
        for i, input_ in enumerate(op.input):
            op.input[i] = canonical_name(input_)
        for i, output in enumerate(op.output):
            op.output[i] = canonical_name(output)


def optimize_interference(net, static_blobs,
                          ordering_function=topological_sort_traversal):
    """
    1) Use a BFS traversal of the execution graph to generate an
       ordering of the node executions.
    2) Generate use-def ranges for each `blob` in the BFS traversal
       order.
    3) Assign blobs to `canonical blobs`
    4) Rename blobs to canonical blobs
    """
    net = copy.deepcopy(net)
    g = compute_interference_graph(net.op)
    ordering = ordering_function(g)
    linearized_ops = [net.op[i] for i in ordering]

    # Reorder ops in net based on the computed linearlized order.
    # If the graph has multiple topological orderings and if the NetDef's
    # ordering differs from the order used to compute ranges, then the
    # runtime might end up overwriting blobs before they are used.
    del net.op[:]
    net.op.extend(linearized_ops)

    ranges = compute_ranges(linearized_ops)
    assignments = compute_assignments(ranges, static_blobs)
    blob_assignments = compute_blob_assignments(assignments)
    apply_assignments(net, blob_assignments)
    return Optimization(
        net=net,
        blob_assignments=blob_assignments,
        assignments=assignments)

Statistics = collections.namedtuple(
    'Statistics', ['baseline_nbytes', 'optimized_nbytes'])


def compute_statistics(assignments):
    def blob_nbytes(blob):
        return workspace.FetchBlob(blob).nbytes
    blob_bytes = {
        blob: blob_nbytes(blob) for assignment in assignments
        for (blob, _) in assignment}
    baseline_nbytes = sum(v for _, v in blob_bytes.iteritems())
    optimized_nbytes = sum(
        max(blob_bytes[blob] for (blob, _) in assignment)
        for assignment in assignments)
    return Statistics(
        baseline_nbytes=baseline_nbytes,
        optimized_nbytes=optimized_nbytes)
