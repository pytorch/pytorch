from collections import defaultdict

from . import ir, scheduler


def _flatten_arg_list(args):
    flat_args = []
    for arg in args:
        if isinstance(arg, (list, tuple)):
            flat_args.extend(_flatten_arg_list(arg))
        else:
            flat_args.append(arg)
    return flat_args


def get_users_from_unfused_nodes(snode):
    # if a fused node has 2 subnodes (A, B) and each subnode has 2 users (A1, A2) and (B1, B2),
    # this function returns (A1, A2, B1, B2).
    if isinstance(snode, scheduler._BaseGroupedSchedulerNode):
        return list(set([user for snode in snode.snodes for user in snode.users]))
    else:
        return snode.users


def get_all_names(snode):
    names = []
    names.append(snode.get_name())
    if isinstance(snode, scheduler._BaseGroupedSchedulerNode):
        for sub_snode in snode.snodes:
            names.extend(get_all_names(sub_snode))
    return names


def get_all_users(snode):
    users = set()
    users.update(snode.users)
    if isinstance(snode, scheduler._BaseGroupedSchedulerNode):
        for sub_snode in snode.snodes:
            users.update(get_all_users(sub_snode))
    return users


def get_all_reads(snode):
    reads = set()
    reads.update(snode.read_writes.reads)
    if isinstance(snode, scheduler._BaseGroupedSchedulerNode):
        for sub_snode in snode.snodes:
            reads.update(get_all_reads(sub_snode))
    return reads


def get_all_writes(snode):
    writes = set()
    writes.update(snode.read_writes.writes)
    if isinstance(snode, scheduler._BaseGroupedSchedulerNode):
        for sub_snode in snode.snodes:
            writes.update(get_all_writes(sub_snode))
    return writes


def collect_node_to_input_prev_writes(snodes):
    # Returns:
    #   snode -> set of node names that write to snode's input args before snode
    write_map = defaultdict(
        set
    )  # bufX -> set of nodes that write to bufX (including itself)
    node_to_input_prev_writes = defaultdict(
        set
    )  # nodeY -> set of writes nodes to nodeY's input args before nodeY
    for snode in snodes:
        for dep in get_all_writes(snode):
            write_map[dep.name].add(snode.get_name())
        if isinstance(snode.node, ir.ResizeStorageBytes):
            write_map[snode.node.resized_buf_name].add(snode.get_name())
        for dep in get_all_reads(snode):
            node_to_input_prev_writes[snode].update(write_map[dep.name])
            node_to_input_prev_writes[snode].discard(snode.get_name())
            if isinstance(snode, scheduler._BaseGroupedSchedulerNode):
                for sub_snode in snode.snodes:
                    node_to_input_prev_writes[sub_snode].update(write_map[dep.name])
                    node_to_input_prev_writes[sub_snode].discard(sub_snode.get_name())
    return node_to_input_prev_writes
