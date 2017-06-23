## @package memonger
# Module caffe2.python.memonger
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import networkx as nx
import collections
import time
import heapq
import copy
from caffe2.python import workspace
from caffe2.proto import caffe2_pb2
import enum
import logging
import numpy as np

log = logging.getLogger("memonger")
log.setLevel(logging.INFO)
LiveRange = collections.namedtuple('LiveRange', ["defined", "used", "size"])


def share_grad_blobs(
    net,
    losses,
    param_grads,
    namescope,
    dont_share_blobs=None,
    share_activations=False,
    blob_shapes=None,
):
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
        for b in list(op.input) + list(op.output):
            if is_grad_blob(b):
                return True
        return False

    log.warn("NOTE: Executing memonger to optimize gradient memory")

    # Collect ops that have something to do with gradients
    if not namescope.endswith("/"):
        namescope += "/"

    netproto = copy.deepcopy(net.Proto())
    activations = []
    external_output = set(net.Proto().external_output)

    # Hacky way to get activations, think of a better way
    for op in net.Proto().op:
        for b in op.output:
            if b + "_w" in op.input and b not in external_output:
                activations.append(b)

    # Remove last activations, as they are usually accessed externally
    activations = set(activations[:-2])

    # Gradient ops
    grad_ops = [op for op in netproto.op if is_grad_op(op)]
    return _compute_blob_recycling_for_dag(
        netproto,
        losses,
        grad_ops,
        lambda b: is_grad_blob(b) or (share_activations and b in activations),
        namescope,
        {} if dont_share_blobs is None else dont_share_blobs,
        blob_shapes
    )


def optimize_inference_for_dag(net, input_blobs, namescope=""):
    netproto = copy.deepcopy(net.Proto())
    external_input = set(net.Proto().external_input)
    external_output = set(net.Proto().external_output)

    def is_activation_blob(b):
        return b not in external_input and b not in external_output

    seen_as_output = set()
    ops = list(net.Proto().op)

    # Sanity check: check that all external inputs are properlyh accounted
    # and that no gradient ops are included in 'net'
    for op in ops:
        for b in op.input:
            if is_activation_blob(b) and b not in seen_as_output:
                assert False, "{} not in external input".format(b)
        seen_as_output = seen_as_output.union(set(op.output))
        assert not op.is_gradient_op, \
            "You can only pass inference-only nets to optimize_inference_for_dag"

    return _compute_blob_recycling_for_dag(
        netproto, input_blobs, ops, is_activation_blob,
        namescope, set(), None,
    )


def _compute_blob_recycling_for_dag(
    netproto, heads, ops, is_shareable,
    namescope, dont_share_blobs, blob_shapes=None,
):
    '''
    Computes a blob recycling by traversing the computation DAG. The resulting
    model can be executed safely on a DAGNet.
    '''
    start_time = time.time()
    if dont_share_blobs is not None:
        dont_share_blobs = set([str(b) for b in dont_share_blobs])

    # Create mapping from blobs to ops
    blobs_to_ops = collections.defaultdict(lambda: [])
    blob_input_count = collections.defaultdict(lambda: 0)
    op_inputs = collections.defaultdict(lambda: 0)
    op_visit_count = collections.defaultdict(lambda: 0)
    share_counts = collections.defaultdict(lambda: 0)
    req_tokens = collections.defaultdict(lambda: set())
    op_token_deposit = [set() for _ in ops]

    blob_sizes = {} if blob_shapes is not None else None

    # First figure out which of the shareable blobs
    # are 'internal' to the optimization. For example, if optimizing
    # only gradient ops, then activation blobs will be 'external' as they
    # are not output by these ops.
    optim_op_outputs = set()
    for op in ops:
        optim_op_outputs.update(set(op.output))

    for i, op in enumerate(ops):
        for inp in op.input:
            if is_shareable(inp) or inp in heads:
                if inp in optim_op_outputs:
                    blobs_to_ops[inp].append(i)
                    op_inputs[i] += 1
                else:
                    # For external blobs, we don't increase the op_inputs
                    # count.
                    blobs_to_ops[inp].append(i)
                    share_counts[inp] = 1

    # Traverse operators starting from the heads' blobs.
    # Keep tabs on when blobs are seen first and last, and also
    # when operators have their input satisfied. Share blobs only
    # under same branch, avoiding problems with parallel workers.
    output_blobs = set()
    mapping = {}
    unknown_shapes = set()

    def infer_blob_size(b):
        if b in blob_shapes:
            return np.prod(blob_shapes[b])
        else:
            unknown_shapes.add(b)
        return 0

    global token_seq
    token_seq = 0

    def next_token():
        global token_seq
        token_seq += 1
        return token_seq

    def compatible(blob, cur_tokens):
        # Do we have all tokens?
        return len(req_tokens[blob] - cur_tokens) == 0

    saved_count = 0

    def descend(op_idx, free_blobs, tokens):
        # Take tokens left here
        tokens = tokens.union(op_token_deposit[op_idx])
        op_token_deposit[op_idx] = None
        cur_op = ops[op_idx]
        new_free_blobs = set()
        unused_free_blobs = set(free_blobs)
        saved = 0

        for b in list(cur_op.input) + list(cur_op.output):
            actual_blob = b if b not in mapping else mapping[b]
            req_tokens[b] = req_tokens[b].union(tokens)
            if actual_blob != b:
                req_tokens[actual_blob] = req_tokens[actual_blob].union(tokens)

        for inp in cur_op.input:
            if is_shareable(inp):
                blob_input_count[inp] += 1
                if blob_input_count[inp] == len(blobs_to_ops[inp]):
                    actual_blob = inp if inp not in mapping else mapping[inp]
                    if actual_blob not in dont_share_blobs:
                        new_free_blobs.add(
                            (-share_counts[actual_blob], actual_blob),
                        )

        for outp in cur_op.output:
            if is_shareable(outp):
                if outp not in output_blobs:
                    # First seen this blob as output, can assign to a free blob
                    freeb = None
                    if blob_sizes is None:
                        put_back = []
                        while len(free_blobs) > 0:
                            (negcnt, cand_freeb) = heapq.heappop(free_blobs)
                            if compatible(cand_freeb, tokens):
                                freeb = cand_freeb
                                break
                            else:
                                put_back.append((negcnt, cand_freeb))
                        for cnt, b in put_back:
                            heapq.heappush(free_blobs, (cnt, b))
                    else:
                        bsize = infer_blob_size(outp)
                        best_blob = None
                        best_size = -1

                        # Heuristic to choose the most suitably sized blob
                        for b in free_blobs:
                            if compatible(b, tokens):
                                sz = blob_sizes[b]
                                if sz >= best_size:
                                    if best_size < bsize or best_size >= sz:
                                        best_size = sz
                                        best_blob = b

                        freeb = best_blob

                        if freeb is not None:
                            free_blobs.remove(freeb)
                            saved += bsize

                    if freeb is not None:
                        req_tokens[freeb] = req_tokens[freeb].union(tokens)
                        mapping[outp] = freeb
                        if freeb in unused_free_blobs:
                            unused_free_blobs.remove(freeb)
                        share_counts[freeb] += 1

                output_blobs.add(outp)

        for (cnt, nf) in new_free_blobs:
            already_inserted = False
            # Note: we prevent double insertion, but it can
            # happen because of parallel branches. Token management
            # ensures free blobs are handled correctly.
            if blob_sizes is None:
                for _c, b in free_blobs:
                    if b == nf:
                        already_inserted = True
                if not already_inserted:
                    heapq.heappush(free_blobs, (cnt, nf))
            else:
                if nf not in blob_sizes:
                    blob_sizes[nf] = infer_blob_size(outp)
                if nf in free_blobs:
                    already_inserted = True
                if not already_inserted:
                    free_blobs.append(nf)

        num_branches = 0
        # Count branches
        for outp in cur_op.output:
            for _ in blobs_to_ops[outp]:
                num_branches += 1

        for outp in cur_op.output:
            for inp_op_idx in blobs_to_ops[outp]:
                op_visit_count[inp_op_idx] += 1

                # Descend only if we have satisfied all inputs
                if op_visit_count[inp_op_idx] == op_inputs[inp_op_idx]:
                    assert inp_op_idx != op_idx
                    new_tokens = tokens
                    if num_branches > 1:
                        # Optimization
                        new_tokens = tokens.union(set([next_token()]))
                    saved_desc = descend(
                        inp_op_idx,
                        free_blobs[:],
                        new_tokens,
                    )
                    saved += saved_desc

                else:
                    # Leave my tokens here
                    if op_token_deposit[inp_op_idx] is not None:
                        op_token_deposit[inp_op_idx] = \
                            op_token_deposit[inp_op_idx].union(tokens)

        return saved

    # Start DFS from the heads' (losses or inputs)
    for head_blob in heads:
        for op_idx in blobs_to_ops[head_blob]:
            saved = descend(op_idx, [], set([next_token()]))
            saved_count += saved

    # Rename the shared blobs
    shared_blobs = set(mapping.values())
    renamed = {}
    for j, b in enumerate(shared_blobs):
        if b in optim_op_outputs:
            renamed[b] = namescope + "__m{}_shared".format(j)
        else:
            renamed[b] = b

    # Update the mapping recursively
    mapping.update(renamed)
    had_changes = True
    while had_changes:
        had_changes = False
        for k, v in mapping.items():
            if v in renamed and renamed[v] != v:
                renamed[k] = renamed[v]
                mapping[k] = renamed[k]
                had_changes = True

    shared_blobs = set(mapping.values())

    if saved_count > 0:
        log.info("Remapping {} blobs, using {} shared; saved apprx {} MB".format(
            len(mapping), len(shared_blobs), int(saved_count * 4 / 1024 / 1024),
        ))
        log.info("Could not infer sizes for: {}".format(unknown_shapes))
    else:
        log.info("Remapping {} blobs, using {} shared".format(
            len(mapping), len(shared_blobs),
        ))

    apply_assignments(netproto, mapping)
    log.info("Memonger memory optimization took {} secs".format(
        time.time() - start_time),
    )
    return netproto


def _find_source_nodes(g):
    ''' Return nodes without predecessors '''
    ret = []
    for cn in g:
        cur_pred = g.predecessors(cn)
        if not cur_pred:
            ret.append(cn)
    return ret


def _find_target_nodes(g):
    ''' Return nodes without successors '''
    ret = []
    for cn in g:
        cur_succ = g.successors(cn)
        if not cur_succ:
            ret.append(cn)
    return ret


def _add_single_target_ifneeded(g):
    targets = _find_target_nodes(g)
    assert len(targets) >= 1
    if len(targets) == 1:
        return g
    ret = copy.deepcopy(g)

    def _next_available_idx(g):
        ret = -1
        for cn in g:
            if cn > ret:
                ret = cn
        ret += 1
        return ret

    target_node_idx = _next_available_idx(g)
    ret.add_node(target_node_idx)
    for cn in targets:
        ret.add_edge(cn, target_node_idx)

    return ret


def _get_path(pred_list, dist_list):
    ''' Get the path from nx.bellman_ford()'s output '''

    # distances are negative
    assert all(dist_list[x] <= 0 for x in dist_list)
    # node with longest distance to source is the target
    target = min(dist_list, key=lambda x: dist_list[x])

    ret = []
    cur = target
    while cur is not None:
        ret.append(cur)
        cur = pred_list[cur]
    return list(reversed(ret))


def _get_longest_paths(g, source_nodes):
    ''' Get the longest path for nodes in 'source_nodes'
        Find with bellman_ford() by setting weight = -1
    '''

    ng = copy.deepcopy(g)
    for u, v in ng.edges():
        ng[u][v]["weight"] = -1

    ret = {}
    for cn in source_nodes:
        pred, dist = nx.bellman_ford(ng, cn, weight="weight")
        path = _get_path(pred, dist)
        assert path[0] == cn
        assert len(path) - 1 == -dist[path[-1]]
        ret[cn] = path

    return ret


def _build_tree(paths):
    ''' Build a tree for given paths based on common elements.
        Last elements of all paths are the same, which is the root of the tree.
    '''
    assert all(cp[-1] == paths[0][-1] for cp in paths)
    g = nx.DiGraph()
    node_set = {y for x in paths for y in x}
    g.add_nodes_from(node_set)
    for cp in paths:
        for ce in zip(cp[0:-1], cp[1:]):
            g.add_edge(ce[1], ce[0])

    root = paths[0][-1]
    _compute_tree_height(g, root)

    return (g, root)


def _compute_tree_height(g, root):
    ''' Compute the heights of the tree for all nodes
        Height of leaves are 0
    '''
    def _get_height(root):
        children = g.successors(root)
        height = 0
        if children:
            child_heights = [_get_height(x) for x in children]
            height = max(child_heights) + 1
        g.node[root]["height"] = height
        return height

    _get_height(root)


def _sort_tree_leaves(g, root):
    ''' For each node, sort its child nodes based on the height of the nodes.
        Return the leaf nodes of the tree after sorting.
    '''
    def _get_height(root):
        return g.node[root]["height"]

    def _get_sorted_leaves(root):
        children = g.successors(root)
        if not children:
            return [root]
        child_heights = [_get_height(x) for x in children]
        order = sorted(range(len(children)), key=lambda x: child_heights[x])
        ret = []
        for co in order:
            cr = children[co]
            ret += _get_sorted_leaves(cr)

        return ret

    return _get_sorted_leaves(root)


def topological_sort_traversal_longest_path(g):
    ''' The graph 'g' may contain several source nodes (nodes without incoming
        edge), which could be in any order and still be a valid
        topological sorting result. We would like to arrange these source nodes
        so that the average live spans of the computed blobs are shorter.
        The idea is to sort the source nodes based on the length of their path to
        the target node so that the one with longer path is used first.
        This is done by:
        - Add a single target node if there are multiple target nodes in 'g'.
        - Find the longest path between each source and the target node.
        - Convert the longest paths to a tree with the target node being the root
          and source nodes being the leaves.
        - Sort the nodes of the tree based on the height of the tree.
    '''
    gt = _add_single_target_ifneeded(g)
    source_nodes = _find_source_nodes(gt)
    lpaths = _get_longest_paths(gt, source_nodes)
    tree, root = _build_tree(lpaths.values())
    sorted_sources = _sort_tree_leaves(tree, root)
    assert(sorted(sorted_sources) == sorted(source_nodes))

    ret = nx.topological_sort(g, sorted_sources)
    assert(len(ret) == len(g.node))
    return ret


def topological_sort_traversal(g):
    return nx.topological_sort(g)


def compute_ranges(linearized_ops, blob_sizes=None):
    if not blob_sizes:
        log.warning('Provide blob sizes to get more accurate assignments.')

    blobs = collections.defaultdict(
        lambda: LiveRange(defined=None, used=None, size=None))
    for i, op in enumerate(linearized_ops):
        for blob in op.input:
            used = blobs[blob].used
            if used is None:
                used = i
            else:
                used = max(used, i)
            blobs[blob] = blobs[blob]._replace(used=used)
            blob_size = blob_sizes[blob] if blob_sizes else None
            assert not blob_sizes or blob_size is not None
            blobs[blob] = blobs[blob]._replace(size=blob_size)
        for blob in op.output:
            defined = blobs[blob].defined
            if defined is None:
                defined = i
            else:
                defined = min(defined, i)
            blobs[blob] = blobs[blob]._replace(defined=defined)
            blob_size = blob_sizes[blob] if blob_sizes else None
            assert not blob_sizes or blob_size is not None
            blobs[blob] = blobs[blob]._replace(size=blob_size)

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


def _get_max_size(assignment):
    if not assignment:
        return 0
    ret = max([x[1].size for x in assignment])
    ret = 0 if ret is None else ret
    return ret


def get_memory_usage(assignments):
    ret = 0
    for cur in assignments:
        ret += _get_max_size(cur)
    return ret


def compute_assignments_greedy(ranges_sorted, init_assignments=None):
    assignments = init_assignments or []
    visited = {y[0] for x in assignments for y in x}

    for (name, range_) in ranges_sorted:
        if name in visited:
            continue
        assigned = False
        best_assignment = 0
        min_dist = float("inf")
        candidate_size = range_.size or 0
        for idx, assignment in enumerate(assignments):
            if is_compatible(range_, assignment, []):
                assigned = True
                dist = abs(_get_max_size(assignment) - candidate_size)
                if dist < min_dist:
                    min_dist = dist
                    best_assignment = idx
        if assigned:
            assignment = assignments[best_assignment]
            assignment.append((name, range_))
        else:
            assignments.append([(name, range_)])
    return assignments


def _get_count(assignments):
    ''' Return number of blobs in assignments '''
    if assignments:
        return sum([len(x) for x in assignments])
    return 0


def compute_assignments_dp(ranges_sorted, init_assignment, counter=None):
    ''' Compute assignment for blobs in 'ranges_sorted' on top of 'init_assignment'
        using dynamic programming + recursion.

        ranges_sorted: blobs sorted by 'used'
        init_assignment: assignment to start with, blobs in 'ranges_sorted' should
                         not be used in 'init_assignment'

        Using f(b, k, init) to represent the best assignment for blobs b[0:k]
        given initial assignment 'init', we have
            f(b, k, init) = f(b, j, init) +
                            find_best(b[j:k], f(b, j, init))
        where j is the index of the last best assignment that is independent of
        blob b[k - 1] (b[k - 1] is compatible with all assignments in
        f(b, j, init)), and find_best(b1, init1) gives the best assignment
        for blobs in 'b1' based on the initial assignment 'init1', and blobs
        b1[0:-1] should be incompatible with b1[-1]. f(b, len(b), []) gives
        the best assignment for blobs 'b'.

        For find_best(b, init), since b[0:-1] are not compatible with b[-1], we
        could reduce it to a smaller problem to find best assignment for b[0:-1]
        as
            find_best(b, init) = min {
                f(b[0:-1], len(b) - 1, init - x) + [x, b[-1]] for x in init, or
                f(b[0:-1], len(b) - 1, init) + [b[-1]]
            }
        where min{} gives the assignment with minimum memory usage.
    '''

    def _get_compatible_prev(candidate_range, best_assignments, cur_idx):
        ''' Find closest position k of best_assignments that is independent of
            candidate_range that candiate_range is compatible with all assignments
            in best_assignments[k].
            Return -1 if not found.
        '''
        def is_compatible_all(candidate_range, assignments):
            ''' return true if compatiable for all assignments in assignments '''
            return all([is_compatible(candidate_range[1], x, []) for x in assignments])

        ii = cur_idx - 1
        while ii >= 0:
            cba = best_assignments[ii]
            if is_compatible_all(candidate_range, cba):
                return ii
            ii -= 1
        return -1

    def _find_best(ranges, init_assignment, prev_best_assignment, counter):
        ''' Find the best assignment for blobs 'ranges' given an initialized
            assignment 'init_assignment'.

            Blobs in ranges[0:-1] should be incompatible with blob range[-1].
            'prev_best_assignment': best assignment for blobs in ranges[:-1]

            By assigning ranges[-1] to each assignment k in 'init_assignment' or
            in a new assignment, the problem becomes a smaller problem to find
            the best assignment for ranges[0:-1] given the initial assignment
            init_assigment[0:k, (k+1):-1].
        '''
        # Blob to check
        find_range = ranges[-1]
        # Blobs in ranges[0:-1] are incompatible with ranges[-1] so that we can
        # reduce it to a smaller problem.
        assert all(not is_compatible(x[1], [find_range], []) for x in ranges[0:-1])

        sz = len(init_assignment)
        best_candidates = []
        # Try to assign 'find_range' to each assignment in init_assignment
        for ii in range(sz):
            if not is_compatible(find_range[1], init_assignment[ii], []):
                continue
            cur_best = copy.deepcopy(init_assignment)
            cur_best[ii].append(find_range)
            if len(ranges) > 1:
                cur_best_tmp = [x for i, x in enumerate(cur_best) if i != ii]
                # reduce to a smaller dp problem
                cur_best_tmp = compute_assignments_dp(
                    ranges[:-1], cur_best_tmp, counter)
                cur_best = cur_best_tmp + [cur_best[ii]]
            best_candidates.append(cur_best)
        # Try to put 'find_range' in a new assignment
        best_candidates.append(prev_best_assignment + [[find_range]])

        ret = min(best_candidates, key=lambda x: get_memory_usage(x))
        return ret

    if not counter:
        counter = [0]
    counter[0] += 1

    if counter and counter[0] % 5000 == 0:
        rs = [ranges_sorted[0][1].defined, ranges_sorted[-1][1].used]
        log.info('Finding assignments {} ({} -> {})...'.format(
            counter[0], rs[0], rs[1]))

    init_assignment = init_assignment or []
    # best_assignments[k]: best assignments for first k blobs ranges_sorted[0:(k+1)]
    best_assignments = []
    # Find best assignment for blobs ranges_sorted[0:ii]
    for ii, cur_range in enumerate(ranges_sorted):
        # closest best_assignment that is independent of ranges_sorted[ii]
        prev_idx = _get_compatible_prev(cur_range, best_assignments, ii)
        prev_best = copy.deepcopy(init_assignment) if prev_idx < 0 else \
                    copy.deepcopy(best_assignments[prev_idx])
        # Need to find best assignment for blobs in 'ranges_part'
        ranges_part = ranges_sorted[(prev_idx + 1):(ii + 1)]
        cur_best = _find_best(
            ranges_part, prev_best,
            best_assignments[-1] if best_assignments else init_assignment,
            counter)
        assert _get_count(cur_best) == _get_count(prev_best) + len(ranges_part)
        best_assignments.append(copy.deepcopy(cur_best))

    assert len(best_assignments) == len(ranges_sorted)

    best = best_assignments[-1]

    return best


def get_updated_ranges(ranges, max_live=None):
    ''' Set LiveRange.defined = -1 if it is None
        Set LiveRange.used = max_live if it is None
        Set LiveRanee.size = 1 if it is None
    '''

    def _get_max_live(ranges):
        max_live = max(x[1].used for x in ranges if x[1].used) + 1
        return max_live

    def _update_range(x, max_live, size):
        cx = x
        if x[1].defined is None:
            cx = (cx[0], cx[1]._replace(defined=-1))
        if x[1].used is None:
            cx = (cx[0], cx[1]._replace(used=max_live))
        if x[1].size is None:
            cx = (cx[0], cx[1]._replace(size=size))
        return cx

    if max_live is None:
        max_live = _get_max_live(ranges)
    ranges = [_update_range(x, max_live, 1) for x in ranges]

    return ranges


def compute_assignments(ranges, static_blobs, algo):
    '''
    algo: Method used to find assignments (AssignmentAlgorithm.GREEDY or
          AssignmentAlgorithm.DYNAMIC_PROGRAMMING).
          AssignmentAlgorithm.DYNAMIC_PROGRAMMING gives optimal solution at the
          cost of more computation.
          AssignmentAlgorithm.GREEDY may be better in the case 'blob_sizes' is
          not provided.
    '''

    # Sort the ranges based on when they are last used.
    # If LiveRange.used is None, then the blob is never used and could
    # be consumed externally. Sort these to the end of the list as opposed
    # to the beginning so that they can be shared as well.
    ranges = sorted(
        list(ranges.items()),
        key=lambda p: (p[1].used is None, p[1].used),
    )
    # Update None values
    ranges = get_updated_ranges(ranges)

    # Sharable blobs
    ranges_sharable = [x for x in ranges if x[0] not in static_blobs]
    # Static blobs, not sharable
    ranges_static = [x for x in ranges if x[0] in static_blobs]

    log.info("Total sharable blobs {}".format(len(ranges_sharable)))

    best_assignment = []
    if algo == AssignmentAlgorithm.DYNAMIC_PROGRAMMING:
        best_assignment = compute_assignments_dp(ranges_sharable, [])
    elif algo == AssignmentAlgorithm.GREEDY:
        best_assignment = compute_assignments_greedy(ranges_sharable, [])
    else:
        assert "Invalid algo name {}".format(algo)
    best_assignment += [[x] for x in ranges_static]

    # verify_assignments(best_assignment)

    return best_assignment


def verify_assignments(assignments):
    for cur in assignments:
        for x, y in zip(cur[0:-1], cur[1:]):
            assert x[1].used < y[1].defined


def compute_interference_graph(ops):
    g = nx.DiGraph()
    for i, op in enumerate(ops):
        g.add_node(i, op=op)
    for i, parent_op in enumerate(ops):
        for j, child_op in enumerate(ops):
            if i >= j:
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
        # Descend into subnets of the recurrent network
        if op.type.startswith('RecurrentNetwork'):
            apply_recurrent_blob_assignments(op, blob_assignments, canonical_name)

        for i, input_ in enumerate(op.input):
            op.input[i] = canonical_name(input_)
        for i, output in enumerate(op.output):
            op.output[i] = canonical_name(output)


def apply_recurrent_blob_assignments(op, blob_assignments, canonical_name):
    log.debug("Applying assignments to recurrent op: {}".format(op.type))
    import google.protobuf.text_format as protobuftx
    step_args = [a for a in op.arg if a.name.endswith("step_net")]
    for step_arg in step_args:
        step_proto = caffe2_pb2.NetDef()
        protobuftx.Merge(step_arg.s.decode("ascii"), step_proto)
        apply_assignments(step_proto, blob_assignments)
        for i, einp in enumerate(step_proto.external_input):
            if einp in blob_assignments:
                step_proto.external_input[i] = canonical_name(einp)
        step_arg.s = str(step_proto).encode("ascii")
    # Store renamings
    for blob, renamed in blob_assignments.items():
        if blob in list(op.input) + list(op.output):
            a = caffe2_pb2.Argument()
            a.name = blob + ".rename"
            a.s = str(renamed).encode("ascii")
            op.arg.extend([a])


class AssignmentAlgorithm(enum.Enum):
    GREEDY = 0
    DYNAMIC_PROGRAMMING = 1


def optimize_interference(net, static_blobs,
                          ordering_function=topological_sort_traversal,
                          blob_sizes=None,
                          algo=AssignmentAlgorithm.GREEDY):
    """
    ordering_function: topological_sort_traversal or
                       topological_sort_traversal_longest_path.
                       topological_sort_traversal_longest_path gives better
                       results but needs a bit more computation.
    algo: Method used to find assignments (AssignmentAlgorithm.GREEDY or
          AssignmentAlgorithm.DYNAMIC_PROGRAMMING).
          AssignmentAlgorithm.DYNAMIC_PROGRAMMING gives optimal solution at the
          cost of more computation.
          AssignmentAlgorithm.GREEDY may be better in the case 'blob_sizes' is
          not provided.
    """

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

    ranges = compute_ranges(linearized_ops, blob_sizes)
    assignments = compute_assignments(ranges, static_blobs, algo)
    blob_assignments = compute_blob_assignments(assignments)
    apply_assignments(net, blob_assignments)
    return Optimization(
        net=net,
        blob_assignments=blob_assignments,
        assignments=assignments)


def verify_graph_equality(net_a, net_b):
    """
    Determines if the execution of two graphs are identical.
    That is, all inputs blobs are mapped to the same output blobs
    for each operator in their respective positions.

    This is meant to check the output of memonger with the original graph.
    It assumes that the nets have same external input and output.
    """

    # Generates "true graph". That is, exactly which inputs will affect which outputs.
    # Two nets are the same if their graphs are the same.
    # For each op, if another op later in execution order overwrites an output blob,
    # it can no longer be considered the parent of future ops through that blob.
    def parent_list(ops):
        # Initialize to be empty for each operator
        parent_list = [[] for _ in ops]
        for i, parent_op in enumerate(ops):
            out_blobs = set(parent_op.output)
            for j, child_op in enumerate(ops):
                if len(out_blobs) == 0:
                    break
                if i >= j:
                    continue
                if any(blob in out_blobs for blob in child_op.input):
                    parent_list[j].append(i)
                for blob in child_op.output:
                    if blob in out_blobs:
                        out_blobs.remove(blob)
        return parent_list

    # Operator wise equality checks
    if (len(net_a.op) != len(net_b.op)):
        return False
    for op_a, op_b in zip(net_a.op, net_b.op):
        if (op_a.type != op_b.type or
                op_a.device_option != op_b.device_option or
                op_a.engine != op_b.engine):
            return False

    # Net wise equality check
    return parent_list(net_a.op) == parent_list(net_b.op)

Statistics = collections.namedtuple(
    'Statistics', ['baseline_nbytes', 'optimized_nbytes'])


def blob_nbytes(blob):
    sz = 0
    try:
        sz = workspace.FetchBlob(blob).nbytes
    except Exception:
        log.warning('Error when fetching blob {}'.format(blob))
    return sz


def compute_statistics(assignments):
    blob_bytes = {
        blob: blob_nbytes(blob) for assignment in assignments
        for (blob, _) in assignment}
    baseline_nbytes = sum(v for _, v in blob_bytes.items())
    optimized_nbytes = sum(
        max(blob_bytes[blob] for (blob, _) in assignment)
        for assignment in assignments)
    return Statistics(
        baseline_nbytes=baseline_nbytes,
        optimized_nbytes=optimized_nbytes)


def collect_blob_sizes(net):
    blobs = {}
    for op in net.op:
        for blob in op.input:
            blobs[blob] = blob_nbytes(blob)
        for blob in op.output:
            blobs[blob] = blob_nbytes(blob)

    return blobs
