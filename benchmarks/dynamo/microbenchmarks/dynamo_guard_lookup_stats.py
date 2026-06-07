import argparse
import json
import os
import statistics
import subprocess
import sys
import time


def fn(x, y, scale: int):
    return (x + y) * scale


def avg_ns(stats, key):
    count = max(int(stats["lookup_count"]), 1)
    return int(stats.get(key, 0)) / count


def child_main(args):
    import torch
    from torch._C._dynamo import guards

    opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
    x = torch.randn(4)
    y = torch.randn(4)

    for _ in range(args.warmup):
        opt_fn(x, y, 3)

    guards.reset_guard_lookup_stats()
    start_ns = time.perf_counter_ns()
    for _ in range(args.iters):
        opt_fn(x, y, 3)
    elapsed_ns = time.perf_counter_ns() - start_ns

    stats = guards.get_guard_lookup_stats()
    accessor_type_stats = {
        str(name): dict(value)
        for name, value in stats["root_guard_accessor_type_stats"].items()
    }
    accessor_detail_topk = {
        str(name): dict(value)
        for name, value in stats["root_guard_accessor_detail_topk"].items()
    }
    payload = {
        "iters": args.iters,
        "call_avg_ns": elapsed_ns / args.iters,
        "lookup_count": stats["lookup_count"],
        "lookup_avg_ns": avg_ns(stats, "lookup_total_ns"),
        "backend_match_avg_ns": avg_ns(stats, "backend_match_ns"),
        "slow_guard_avg_ns": avg_ns(stats, "slow_guard_ns"),
        "move_to_front_avg_ns": avg_ns(stats, "move_to_front_ns"),
        "root_guard_avg_ns": avg_ns(stats, "root_guard_total_ns"),
        "root_guard_lock_avg_ns": avg_ns(stats, "root_guard_lock_ns"),
        "root_guard_local_state_avg_ns": avg_ns(
            stats, "root_guard_local_state_ns"
        ),
        "root_guard_leaf_avg_ns": avg_ns(stats, "root_guard_leaf_ns"),
        "root_guard_accessor_avg_ns": avg_ns(stats, "root_guard_accessor_ns"),
        "root_guard_tls_avg_ns": avg_ns(stats, "root_guard_tls_ns"),
        "root_guard_count": stats["root_guard_count"],
        "cache_entry_count_sum": stats["cache_entry_count_sum"],
        "cache_entry_hit_index_sum": stats["cache_entry_hit_index_sum"],
        "root_guard_accessor_type_stats": accessor_type_stats,
        "root_guard_accessor_detail_topk": accessor_detail_topk,
        "unsafe_mock_guard_bypass_enabled": bool(
            stats["unsafe_mock_guard_bypass_enabled"]
        ),
        "unsafe_mock_guard_bypass_count": stats["unsafe_mock_guard_bypass_count"],
        "unsafe_mock_guard_bypass_hit_index_sum": stats[
            "unsafe_mock_guard_bypass_hit_index_sum"
        ],
    }
    print(json.dumps(payload, sort_keys=True))


def run_once(iters, warmup):
    env = os.environ.copy()
    env["TORCHDYNAMO_GUARD_LOOKUP_STATS"] = "1"
    if os.environ.get("TORCHDYNAMO_UNSAFE_MOCK_GUARD_BYPASS") == "1":
        env["TORCHDYNAMO_UNSAFE_MOCK_GUARD_BYPASS"] = "1"
    output = subprocess.check_output(
        [
            sys.executable,
            __file__,
            "--child",
            "--iters",
            str(iters),
            "--warmup",
            str(warmup),
        ],
        env=env,
        text=True,
    )
    return json.loads(output.splitlines()[-1])


def median(rows, key):
    return statistics.median(float(row[key]) for row in rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=10000)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args()

    if args.child:
        child_main(args)
        return

    rows = [run_once(args.iters, args.warmup) for _ in range(args.repeats)]

    columns = [
        "call_us",
        "lookup_us",
        "backend_us",
        "slow_guard_us",
        "move_to_front_us",
        "root_guard_us",
        "root_lock_us",
        "root_local_state_us",
        "root_leaf_us",
        "root_accessor_us",
        "root_tls_us",
        "lookup_count",
        "root_guard_count",
        "mock_bypass_count",
    ]
    values = [
        f"{median(rows, 'call_avg_ns') / 1000:.3f}",
        f"{median(rows, 'lookup_avg_ns') / 1000:.3f}",
        f"{median(rows, 'backend_match_avg_ns') / 1000:.3f}",
        f"{median(rows, 'slow_guard_avg_ns') / 1000:.3f}",
        f"{median(rows, 'move_to_front_avg_ns') / 1000:.3f}",
        f"{median(rows, 'root_guard_avg_ns') / 1000:.3f}",
        f"{median(rows, 'root_guard_lock_avg_ns') / 1000:.3f}",
        f"{median(rows, 'root_guard_local_state_avg_ns') / 1000:.3f}",
        f"{median(rows, 'root_guard_leaf_avg_ns') / 1000:.3f}",
        f"{median(rows, 'root_guard_accessor_avg_ns') / 1000:.3f}",
        f"{median(rows, 'root_guard_tls_avg_ns') / 1000:.3f}",
        str(int(median(rows, "lookup_count"))),
        str(int(median(rows, "root_guard_count"))),
        str(int(median(rows, "unsafe_mock_guard_bypass_count"))),
    ]
    print(" ".join(columns))
    print(" ".join(values))
    accessor_totals = {}
    for row in rows:
        for name, item in row["root_guard_accessor_type_stats"].items():
            total = accessor_totals.setdefault(
                name, {"count": 0, "fail_count": 0, "inclusive_ns": 0}
            )
            total["count"] += int(item["count"])
            total["fail_count"] += int(item["fail_count"])
            total["inclusive_ns"] += int(item["inclusive_ns"])

    print("\naccessor_type_stats:")
    print("name count fail_count inclusive_us avg_inclusive_us")
    for name, item in sorted(
        accessor_totals.items(),
        key=lambda pair: pair[1]["inclusive_ns"],
        reverse=True,
    ):
        count = max(item["count"], 1)
        print(
            " ".join(
                [
                    name,
                    str(item["count"]),
                    str(item["fail_count"]),
                    f"{item['inclusive_ns'] / 1000:.3f}",
                    f"{item['inclusive_ns'] / count / 1000:.3f}",
                ]
            )
        )
    print("\naccessor_detail_topk:")
    print("name count fail_count self_us child_us inclusive_us avg_inclusive_us")
    detail_totals = {}
    for row in rows:
        for name, item in row["root_guard_accessor_detail_topk"].items():
            total = detail_totals.setdefault(
                name,
                {
                    "count": 0,
                    "fail_count": 0,
                    "self_ns": 0,
                    "child_ns": 0,
                    "inclusive_ns": 0,
                },
            )
            total["count"] += int(item["count"])
            total["fail_count"] += int(item["fail_count"])
            total["self_ns"] += int(item["self_ns"])
            total["child_ns"] += int(item["child_ns"])
            total["inclusive_ns"] += int(item["inclusive_ns"])
    for name, item in sorted(
        detail_totals.items(),
        key=lambda pair: pair[1]["inclusive_ns"],
        reverse=True,
    ):
        count = max(item["count"], 1)
        print(
            " ".join(
                [
                    name,
                    str(item["count"]),
                    str(item["fail_count"]),
                    f"{item['self_ns'] / 1000:.3f}",
                    f"{item['child_ns'] / 1000:.3f}",
                    f"{item['inclusive_ns'] / 1000:.3f}",
                    f"{item['inclusive_ns'] / count / 1000:.3f}",
                ]
            )
        )

    print("\nraw_json:")
    print(json.dumps(rows, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
