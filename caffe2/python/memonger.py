from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import networkx as nx
import collections
import copy
from caffe2.python import workspace

LiveRange = collections.namedtuple('LiveRange', ["defined", "used"])


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
    if candidate_range.used is None or candidate_range.defined is None \
       or range_.defined is None or range_.used is None:
        return False
    return candidate_range.defined > range_.used


def compute_blob_assignments(assignments):
    blob_assignments = {}
    for assignment in assignments:
        if len(assignment) == 1:
            continue
        for (blob, _) in assignment:
            blob_assignments[blob] = ",".join([b for b, _ in assignment])
    return blob_assignments


def compute_assignments(ranges, static_blobs):
    ranges = sorted(list(ranges.iteritems()), key=lambda p: p[1].used)
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
        return b"{}_shared".format(blob_assignments[blob])

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
