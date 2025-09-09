import collections
import logging
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Optional, TypeVar, Union

import torch
import torch.distributed as dist
import torch.utils._pytree as pytree
from torch._dispatch.python import enable_python_dispatcher
from torch._dynamo.utils import detect_fake_mode
from torch._inductor.runtime.runtime_utils import dynamo_timed
from torch._logging import trace_structured
from torch.fx.experimental.proxy_tensor import make_fx
from torch.utils._ordered_set import OrderedSet

from .bucketing import collect_node_descendents, merge_all_gather, merge_reduce_scatter


logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class CType(str, Enum):
    AG = "AG"  # "all_gather"
    AGC = "AGC"  # "all_gather_coalesced"
    AR = "AR"  # "all_reduce"
    ARC = "ARC"  # "all_reduce_coalesced"
    RS = "RS"  # "reduce_scatter"
    RSC = "RSC"  # "reduce_scatter_coalesced"
    A2A = "A2A"  # "all_2_all"
    CMP = "C"  # "compute"


@dataclass
class CKey:
    ctype: CType
    group_name: Optional[str] = None

    def __hash__(self):
        return hash((self.ctype, self.group_name))

    def __str__(self):
        return f"{self.ctype}({self.group_name})"


def get_collective_key(n: torch.fx.Node) -> CKey:
    if n.op != "call_function":
        return CKey(CType.CMP, None)
    if n.target == torch.ops._c10d_functional.all_gather_into_tensor.default:
        _, group_size, group_name = n.args
        assert isinstance(group_name, str)
        return CKey(CType.AG, group_name)
    elif (
        n.target == torch.ops._c10d_functional.all_gather_into_tensor_coalesced.default
    ):
        assert False
        return CKey(CType.AGC, None)
    if n.target == torch.ops._c10d_functional.all_reduce.default:
        # TODO: Verify args
        _, _, group_name = n.args
        assert isinstance(group_name, str)
        return CKey(CType.AR, group_name)
    elif n.target == torch.ops._c10d_functional.all_reduce_coalesced.default:
        assert False
        return CKey(CType.ARC, None)
    elif n.target == torch.ops._c10d_functional.reduce_scatter_tensor.default:
        _, reduce_op, group_size, group_name = n.args
        assert isinstance(group_name, str)
        return CKey(CType.RS, group_name)
    elif n.target == torch.ops._c10d_functional.reduce_scatter_tensor_coalesced.default:
        assert False
        return CKey(CType.RSC, None)
    elif n.target == torch.ops._c10d_functional.all_to_all_single.default:
        assert False
        return CKey(CType.A2A, None)
    return CKey(CType.CMP, None)


class CTreeNode:
    def __init__(self, ctype, group_name, parent_ct=None):
        self.ctype = ctype
        self.group_name = group_name
        self.parent_ct = parent_ct
        self.depth = 0 if self.parent_ct is None else parent_ct.depth + 1
        self.n_to_parent_n = {}
        self.n_to_bucket: dict[torch.fx.Node, Optional[str]] = {}
        self.children_cts = {}  # dict[CKey, CTreeNode]


def get_collectives_trie(
    gm: torch.fx.GraphModule,
):
    g = gm.graph
    n_to_ct = {}
    collectives_set = OrderedSet()
    root_cts = {}
    for n in g.nodes:
        ckey = get_collective_key(n)
        ctype, group_name = ckey.ctype, ckey.group_name

        args_classes = []
        ct_parent_n = None
        args_min_depth = len(g.nodes)
        args_min_depth_ct = None
        for arg in n.args:
            if not isinstance(arg, torch.fx.Node):
                continue

            arg_ct = n_to_ct.get(arg, None)
            if arg_ct is None:
                continue
            if arg_ct.depth < args_min_depth:
                args_min_depth = arg_ct.depth
                args_min_depth_ct = arg_ct
                ct_parent_n = arg

        args_ct = args_min_depth_ct

        if ckey.ctype == CType.CMP:
            if args_ct is not None:
                n_to_ct[n] = args_ct
            continue

        lookup_cts = args_ct.children_cts if args_ct is not None else root_cts
        ct = lookup_cts.get(ckey, None)
        if ct is not None:
            ct.n_to_parent_n[n] = ct_parent_n
            n_to_ct[n] = args_ct
            continue

        ct = CTreeNode(
            ctype,
            group_name,
            parent_ct=args_ct,
        )
        ct.n_to_parent_n[n] = ct_parent_n
        if args_ct is None:
            root_cts[ckey] = ct
        else:
            args_ct.children_cts[ckey] = ct

        n_to_ct[n] = ct

    def _dfs(ct, curr_path):
        path = curr_path
        if path:
            path += "-"
        path += f"{ct.ctype}({ct.group_name})"
        for _ct in ct.children_cts.values():
            _dfs(_ct, path)

    for ct in root_cts.values():
        _dfs(ct, "")

    return root_cts


def _get_nn_module_stack_path(n) -> str:
    nn_module_stack = n.meta.get("nn_module_stack", None)
    if nn_module_stack is None:
        return ""

    for key, val in nn_module_stack.items():
        if "layers" in key:
            s = val[0]
            layer_num = int(val[0].split(".")[1])
            layer_group = layer_num // 32
            return f"layer_group_{layer_group}"
    return ""


def _greedy_bucket(
    ns,
    bucket_cap_mb_by_bucket_idx: Callable[[int], float],
    node_descendents,
    log,
    try_left=3,
):
    if len(ns) <= 1:
        return []

    buckets: list[list[torch.fx.Node]] = []
    cur_bucket: list[torch.fx.Node] = []
    cur_bucket_descendents: OrderedSet[torch.fx.Node] = OrderedSet()
    cur_bucket_size_bytes: int = 0
    cur_bucket_id: int = 0
    bucket_size_bytes = int(bucket_cap_mb_by_bucket_idx(cur_bucket_id) * 1024 * 1024)
    left_ns = []

    def _add_bucket(bucket_ns, bucket_descendents):
        if len(bucket_ns) <= 1:
            log.append(f"\n cur_bucket <= 1 {bucket_ns} SKIP")
            return False
        # sorted_bucket_ns = sorted(bucket_ns, key=lambda n: -len(node_descendents[n]))
        # sorted_bucket_ns = bucket_ns
        # d = {n: -len(node_descendents[n]) for n in sorted_bucket_ns}
        # log += f"\n _ADD_BUCKET(sorted:{d})"
        buckets.append(bucket_ns)
        # bucket_ns_set = OrderedSet(bucket_ns)
        # print(f"XXX ADD_BUCKET:{bucket_ns}")
        # print(f"XXX BUCKET_DESC:{bucket_descendents}")
        # for n, descendents in node_descendents.items():
        #     if n in bucket_ns or descendents & bucket_ns_set:
        #         node_descendents[n] |= bucket_descendents
        return True

    # We want to maximize number of bucketed nodes
    sorted_ns = sorted(ns, key=lambda n: len(node_descendents[n]))
    d = {n: len(node_descendents[n]) for n in sorted_ns}
    log.append(f"\n SORTED:{d}")
    log.append(f"\n sorted_ns: {sorted_ns}")
    for node in sorted_ns:
        node_desc = node_descendents[node]
        if node in cur_bucket_descendents or any(bn in node_desc for bn in cur_bucket):
            log.append(
                f"\n {node} SKIP, in cur_bucket_descendents for cur_bucket:{cur_bucket}"
            )
            # if there is a path from node to the current bucket, we cannot horizontally fuse (bucket)
            left_ns.append(node)
            continue
        assert "val" in node.meta
        n_val = node.meta["val"]
        out_size_bytes = n_val.numel() * n_val.element_size()
        n_input_val = node.all_input_nodes[0].meta["val"]
        in_size_bytes = n_input_val.numel() * n_input_val.element_size()
        size_bytes = max(out_size_bytes, in_size_bytes)
        if cur_bucket_size_bytes + size_bytes > bucket_size_bytes and cur_bucket:
            # Current bucket is full, create new bucket
            if not _add_bucket(cur_bucket, cur_bucket_descendents):
                left_ns.extend(cur_bucket)
            cur_bucket = []
            cur_bucket_size_bytes = 0
            cur_bucket_id += 1
            cur_bucket_descendents = OrderedSet()
        cur_bucket_size_bytes += size_bytes
        # print(f"XXX CUR_BUCKET_BEFORE[{node}]:{cur_bucket}")
        cur_bucket.append(node)
        log.append(f"\nXXX CUR_BUCKET++:{cur_bucket}")
        # print(f"XXX CUR_BUCKET_DESC_BEFORE[{node}]:{cur_bucket_descendents}")
        # print(f"XXX CUR_BUCKET_APPEND node:{node} DESCENDENTS:{node_descendents[node]}")
        cur_bucket_descendents |= node_descendents[node]
        # print(f"XXX CUR_BUCKET_DESC_ADD[{node}]:{cur_bucket_descendents}")
        cur_bucket_set = OrderedSet(cur_bucket)
        for n, descendents in node_descendents.items():
            if n in cur_bucket_set or descendents & cur_bucket_set:
                node_descendents[n] |= cur_bucket_descendents
    if not _add_bucket(cur_bucket, cur_bucket_descendents):
        left_ns.extend(cur_bucket)
    # Nothing was bucketed
    if len(buckets) == 0:
        return _greedy_bucket(
            ns[:-1],
            bucket_cap_mb_by_bucket_idx,
            node_descendents,
            log,
            try_left,
        )

    # Something was bucketed
    # Try to bucket the rest.
    if try_left > 0 and len(left_ns) > 1:
        ns_to_bucket = left_ns
        # first one should become last if all others are descendents
        left_buckets = _greedy_bucket(
            ns_to_bucket,
            bucket_cap_mb_by_bucket_idx,
            node_descendents,
            log,
            try_left=try_left - 1,
        )
        buckets.extend(left_buckets)
    return buckets


def bucket_collectives_trie(gm, config):
    g_str = ""
    for n in gm.graph.nodes:
        g_str += f"\n node:{n} {n.op} {n.target}"
        if "nn_module_stack" in n.meta:
            nnms = n.meta["nn_module_stack"]
            g_str += f"\n nn_module_stack:{nnms}"
    trace_structured(
        "artifact",
        metadata_fn=lambda: {
            "name": "bucketing_fx_pre_trie_graph_nn_module_stack",
            "encoding": "string",
        },
        payload_fn=lambda: g_str,
    )
    node_descendents = collect_node_descendents(gm.graph)
    root_cts = get_collectives_trie(gm)
    log = []

    def should_bucket(ctype):
        if ctype == CType.AG:
            return "ag" in config
        elif ctype == CType.RS:
            return "rs" in config
        elif ctype == CType.AR:
            return "ar" in config
        return False

    # TODO: put in config
    tp = 4
    bucket_cap_mb_by_bucket_idx: Callable[[int], float] = lambda id: 2000

    final_buckets = defaultdict(list)

    def _n_in_ct_str(n, ct):
        bucket = ct.n_to_bucket.get(n, "")
        return f"{ct.ctype}({ct.group_name})[{bucket}]"

    def _n_in_ct_path(n, ct):
        cur_ct = ct
        cur_n = n
        ret = ""
        while cur_ct is not None:
            path = _n_in_ct_str(n, ct)
            if ret:
                ret = "-" + ret
            ret = path + ret
            cur_n = cur_ct.n_to_parent_n.get(cur_n, None)
            cur_ct = cur_ct.parent_ct
            if cur_ct is None:
                break
        return ret

    def bucket_ct(ct):
        # group first nodes in ct by full path, including parent cts buckets
        ns_groups = defaultdict(list)
        for n, parent_n in ct.n_to_parent_n.items():
            path = _n_in_ct_path(n, ct)
            dtype = n.meta["val"].dtype
            nn_module_stack_path = ""
            if ct.ctype == CType.AG:
                # Do not factorize RS by layers
                if torch._inductor.config.bucket_fx_collectives_trie_use_nn_module_stack:
                    nn_module_stack_path = _get_nn_module_stack_path(n)

            # group_key = (path, nn_module_stack_path)
            def _dtype_key(dtype):
                if dtype == torch.float32 or dtype == torch.bfloat16:
                    return "bf16_fp32_group"
                return dtype

            dtype_key = _dtype_key(dtype)
            group_key = (path, dtype_key, nn_module_stack_path)
            ns_groups[group_key].append(n)

        ret_buckets = []
        for group_key, ns_group in ns_groups.items():
            log.append(
                f"\n \nBUCKETING KEY:{group_key} {len(ns_group)} -> GROUP:{ns_group}"
            )
            g_buckets = _greedy_bucket(
                ns_group,
                bucket_cap_mb_by_bucket_idx,
                node_descendents,
                log,
                try_left=16,
            )

            log.append(
                f"\n BUCKETING KEY: {group_key} {len(ns_group)} -> G_BUCKETS:{g_buckets}"
            )
            total_ns = len(ns_group)
            total_bucketed = 0
            for bucket_idx, g_bucket in enumerate(g_buckets):
                log.append(
                    f"\n {group_key} -> BUCKET[{bucket_idx}]{len(g_bucket)}: {g_bucket}"
                )
                for g_n in g_bucket:
                    ct.n_to_bucket[g_n] = bucket_idx
                total_bucketed += len(g_bucket)
            perc = total_bucketed / total_ns
            log.append(
                f"\n BUCKETING_RESULT {group_key} bucketed:{total_bucketed} total_ns:{total_ns} percent:{perc}"
            )
            ret_buckets.extend(g_buckets)
        return ret_buckets

    def _dfs(ct, curr_path):
        path = curr_path
        if path:
            path += "-"
        path += f"{ct.ctype}({ct.group_name})"
        log.append(
            f"\n\n CTreeNode[{path}] ctype:{ct.ctype}   ns:{list(ct.n_to_parent_n.keys())}"
        )
        if should_bucket(ct.ctype):
            bs = bucket_ct(ct)
            final_buckets[ct.ctype].extend(bs)
            log.append(f"\n TRIE_BUCKET {path} {ct.ctype}: BUCKETS:{bs}")

        for _ct in ct.children_cts.values():
            _dfs(_ct, path)

    for ct in root_cts.values():
        _dfs(ct, "")

    log.append(f"\n FINAL_BUCKETS:{final_buckets}")

    ag_buckets = []
    rs_buckets = []
    for ctype, ctype_bs in final_buckets.items():
        log.append(f"\n FINAL_BUCKET[{ctype}]:{ctype_bs}")
        if ctype == CType.AG:
            ag_buckets.extend(ctype_bs)
        elif ctype == CType.RS:
            rs_buckets.extend(ctype_bs)
        else:
            assert False

    total_num_ag_rs = 0
    bucketed_ag_rs = OrderedSet()
    log.append(f"\n NUM_AG_BUCKETS:{len(ag_buckets)}: {ag_buckets}")
    log.append(f"\n NUM_RS_BUCKETS:{len(rs_buckets)}: {rs_buckets}")
    for i, ag_bucket in enumerate(ag_buckets):
        log.append(f"\n AG_BUCKET[{i}]:{len(ag_bucket)} nodes:{ag_bucket}")
        for n in ag_bucket:
            bucketed_ag_rs.add(n)
    for n in gm.graph.nodes:
        ckey = get_collective_key(n)
        if ckey.ctype == CType.AG or ckey.ctype == CType.RS:
            total_num_ag_rs += 1

    # for n in gm.graph.nodes:
    #     if is_all_gather_into_tensor(n) or is_reduce_scatter_tensor(n):
    #         total_num_ag_rs += 1
    merge_all_gather(gm, ag_buckets, "custom_ops_multidtype")
    for i, rs_bucket in enumerate(rs_buckets):
        log.append(f"\n RS_BUCKET[{i}]:{len(rs_bucket)} nodes:{rs_bucket}")
        for n in rs_bucket:
            bucketed_ag_rs.add(n)
    merge_reduce_scatter(gm, rs_buckets, "custom_ops_multidtype")
    rs_buckets = bucket_reduce_scatter_all(
        gm, bucket_cap_mb_by_bucket_idx, "custom_ops_multidtype"
    )
    for rs_bucket in rs_buckets:
        for n in rs_bucket:
            bucketed_ag_rs.add(n)

    num_bucketed_ag_rs = len(bucketed_ag_rs)
    percent = -1
    if total_num_ag_rs != 0:
        percent = num_bucketed_ag_rs / total_num_ag_rs
    log.append(
        f"\nXXX TOTAL_NUM_AG_RS:{total_num_ag_rs} BUCKETED:{num_bucketed_ag_rs} PERCENT:{percent}"
    )
    trace_structured(
        "artifact",
        metadata_fn=lambda: {
            "name": "fx_bucketing_collectives_TRIE",
            "encoding": "string",
        },
        payload_fn=lambda: "".join(log),
    )


def __bucket_collectives_trie(gm, config):
    print(f"XXX __BUCKET_COLLECTIVES_TRIE")
    bucket_cap_mb_by_bucket_idx: Callable[[int], float] = lambda id: 2000
    bucket_all_gather_all(gm, bucket_cap_mb_by_bucket_idx, "custom_ops_multidtype")
    bucket_reduce_scatter_all(gm, bucket_cap_mb_by_bucket_idx, "custom_ops_multidtype")


def bucket_all_gather_all(
    gm: torch.fx.GraphModule,
    bucket_cap_mb_by_bucket_idx: Optional[Callable[[int], float]] = None,
    mode: Optional[str] = None,
) -> None:
    if bucket_cap_mb_by_bucket_idx is None:
        from torch._inductor.fx_passes.bucketing import (
            bucket_cap_mb_by_bucket_idx_default,
        )

        bucket_cap_mb_by_bucket_idx = bucket_cap_mb_by_bucket_idx_default
    ag_buckets = bucket_all_gather_by_mb_all(
        gm, bucket_cap_mb_by_bucket_idx, None, mode
    )
    if len(ag_buckets) == 0:
        return
    merge_all_gather(gm, ag_buckets, mode)


def bucket_reduce_scatter_all(
    gm: torch.fx.GraphModule,
    bucket_cap_mb_by_bucket_idx: Optional[Callable[[int], float]] = None,
    mode: Optional[str] = None,
) -> None:
    if bucket_cap_mb_by_bucket_idx is None:
        from torch._inductor.fx_passes.bucketing import (
            bucket_cap_mb_by_bucket_idx_default,
        )

        bucket_cap_mb_by_bucket_idx = bucket_cap_mb_by_bucket_idx_default
    rs_buckets = bucket_reduce_scatter_by_mb_all(
        gm, bucket_cap_mb_by_bucket_idx, None, mode
    )
    if len(rs_buckets) == 0:
        return rs_buckets
    merge_reduce_scatter(gm, rs_buckets, mode)
    return rs_buckets


def bucket_all_gather_by_mb_all(
    gm: torch.fx.GraphModule,
    bucket_cap_mb_by_bucket_idx: Callable[[int], float],
    filter_wait_node: Optional[Callable[[torch.fx.Node], bool]] = None,
    mode: Optional[str] = None,
) -> list[list[torch.fx.Node]]:
    def _ag_group_key(node: torch.fx.Node) -> tuple[str, torch.dtype]:
        _, group_size, group_name = node.args
        dtype = node.meta["val"].dtype
        assert isinstance(group_name, str)
        return (group_name, dtype)

    def _ag_group_key_multidtype(node: torch.fx.Node) -> tuple[str, torch.dtype]:
        _, group_size, group_name = node.args
        assert isinstance(group_name, str)
        return group_name

    group_key_fn = (
        _ag_group_key_multidtype if mode and "multidtype" in mode else _ag_group_key
    )

    return greedy_bucket_collective_by_mb_all(
        gm,
        bucket_cap_mb_by_bucket_idx,
        is_all_gather_into_tensor,
        group_key_fn,
        filter_wait_node,
    )


def bucket_reduce_scatter_by_mb_all(
    gm: torch.fx.GraphModule,
    bucket_cap_mb_by_bucket_idx: Callable[[int], float],
    filter_wait_node: Optional[Callable[[torch.fx.Node], bool]] = None,
    mode: Optional[str] = None,
) -> list[list[torch.fx.Node]]:
    def _rs_group_key(node: torch.fx.Node) -> tuple[str, str, torch.dtype]:
        _, reduce_op, group_size, group_name = node.args
        dtype = node.meta["val"].dtype
        assert isinstance(group_name, str)
        assert isinstance(reduce_op, str)
        return (group_name, reduce_op, dtype)

    def _rs_group_key_multidtype(node: torch.fx.Node) -> tuple[str, str, torch.dtype]:
        _, reduce_op, group_size, group_name = node.args
        assert isinstance(group_name, str)
        assert isinstance(reduce_op, str)
        return (group_name, reduce_op)

    group_key_fn = (
        _rs_group_key_multidtype if mode and "multidtype" in mode else _rs_group_key
    )

    return greedy_bucket_collective_by_mb_all(
        gm,
        bucket_cap_mb_by_bucket_idx,
        is_reduce_scatter_tensor,
        group_key_fn,
        filter_wait_node,
    )


def greedy_bucket_collective_by_mb_all(
    gm: torch.fx.GraphModule,
    bucket_cap_mb_by_bucket_idx: Callable[[int], float],
    filter_node: Callable[[torch.fx.Node], bool],
    node_group_key: Callable[[torch.fx.Node], Any],
    filter_wait_node: Optional[Callable[[torch.fx.Node], bool]] = None,
) -> list[list[torch.fx.Node]]:
    g = gm.graph
    found_candidates = False
    for node in g.nodes:
        if filter_node(node):
            found_candidates = True
            break
    if not found_candidates:
        return []

    # TODO: pearce kelly algorithm for detecting cycles
    log = ""

    nodes_groups: dict[Any, list[torch.fx.Node]] = defaultdict(list)

    for node in g.nodes:
        if is_wait_tensor(node) and filter_node(node.args[0]):
            if (filter_wait_node is None) or filter_wait_node(node):
                coll_node = node.args[0]
                group_key = node_group_key(coll_node)
                nodes_groups[group_key].append(coll_node)

    node_descendents = collect_node_descendents(gm.graph)
    log = []
    for group_key, ns_group in nodes_groups.items():
        log.append(
            f"\n \nBUCKETING KEY:{group_key} {len(ns_group)} -> GROUP:{ns_group}"
        )
        buckets = _greedy_bucket(
            ns_group, bucket_cap_mb_by_bucket_idx, node_descendents, log, try_left=16
        )
    num_total = sum(len(ns) for ns in nodes_groups.values())
    num_bucketed = sum(len(bucket) for bucket in buckets)
    log.append(
        f"\n XXX TOTAL:{num_total} num_bucketed:{num_bucketed} perc:{num_bucketed / num_total}"
    )
    trace_structured(
        "artifact",
        metadata_fn=lambda: {
            "name": "fx_bucketing_greedy_bucket_collective_by_mb_ALL",
            "encoding": "string",
        },
        payload_fn=lambda: "".join(log),
    )
    return buckets

    # buckets: list[list[torch.fx.Node]] = []
    # for group_key, _nodes in nodes_groups.items():
    #     node_descendents = collect_node_descendents(gm.graph)
    #     nodes = sorted(_nodes, key=lambda n: len(node_descendents[n]))
    #     log += f"\n GROUP[{group_key}]: {nodes}"
    #     cur_bucket: list[torch.fx.Node] = []
    #     cur_bucket_descendents: OrderedSet[torch.fx.Node] = OrderedSet()
    #     cur_bucket_size_bytes: int = 0
    #     cur_bucket_id: int = 0
    #     bucket_size_bytes = int(
    #         bucket_cap_mb_by_bucket_idx(cur_bucket_id) * 1024 * 1024
    #     )

    #     def _add_bucket(bucket_ns, bucket_descendents):
    #         nonlocal node_descendents
    #         if len(cur_bucket) <= 1:
    #             nonlocal log
    #             log += f"\n cur_bucket <= 1 {cur_bucket} SKIP"
    #             return False
    #         buckets.append(bucket_ns)
    #         bucket_ns_set = OrderedSet(bucket_ns)
    #         for n, descendents in node_descendents.items():
    #             if (n in bucket_ns_set) or (descendents & bucket_ns_set):
    #                 node_descendents[n] |= bucket_descendents
    #         return True

    #     for node in nodes:
    #         node_desc = node_descendents[node]
    #         bucket_n_in_node_desc = True
    #         for bucket_n in cur_bucket:
    #             if bucket_n in node_desc:
    #                 log += (
    #                     f"\ni {node} SKIP BUCKET_NODE {bucket_n} in NODE_DESC"
    #                 )
    #                 bucket_n_in_node_desc = True
    #                 break
    #         if bucket_n_in_node_desc:
    #             break
    #         if node in cur_bucket_descendents:
    #             # if there is a path from node to the current bucket, we cannot horizontally fuse (bucket)
    #             log += (
    #                 f"\n{node} SKIP in cur_bucket_descendents cur_bucket:{cur_bucket}"
    #             )
    #             continue
    #         assert "val" in node.meta
    #         n_val = node.meta["val"]
    #         out_size_bytes = n_val.numel() * n_val.element_size()
    #         n_input_val = node.all_input_nodes[0].meta["val"]
    #         in_size_bytes = n_input_val.numel() * n_input_val.element_size()
    #         size_bytes = max(out_size_bytes, in_size_bytes)
    #         if cur_bucket_size_bytes + size_bytes > bucket_size_bytes and cur_bucket:
    #             # Current bucket is full, create new bucket
    #             _add_bucket(cur_bucket, cur_bucket_descendents)
    #             cur_bucket = []
    #             cur_bucket_size_bytes = 0
    #             cur_bucket_id += 1
    #             cur_bucket_descendents = OrderedSet()
    #         cur_bucket_size_bytes += size_bytes
    #         cur_bucket.append(node)
    #         cur_bucket_descendents |= node_descendents[node]
    #         cur_bucket_set = set(cur_bucket)
    #         for n, descendents in node_descendents.items():
    #             if (n in cur_bucket_set) or (descendents & cur_bucket_set):
    #                 node_descendents[n] |= cur_bucket_descendents

    #     _add_bucket(cur_bucket, cur_bucket_descendents)
    # log += f"\nBUCKETS:{str(buckets)}"
    # trace_structured(
    #     "artifact",
    #     metadata_fn=lambda: {
    #         "name": "fx_bucketing_greedy_bucket_collective_by_mb_ALL",
    #         "encoding": "string",
    #     },
    #     payload_fn=lambda: log,
    # )
    # return buckets
