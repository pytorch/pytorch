import collections
from multiprocessing import Pool
import ast
import json
import os
from collections import defaultdict, Counter, OrderedDict
from pprint import pprint, pformat

import matplotlib.patches as patches
import numpy as np
from matplotlib import pyplot as plt

from memory_allocator.ormip import solve_cp


def filter_name(name):
    filtered_out_names = [
        "profiler::_record_function_enter",
        "profiler::_record_function_exit",
        "aten::is_leaf",
        "aten::output_nr",
        "aten::_version",
    ]
    return name in filtered_out_names


def get_cmap(n, name="spring"):
    """Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name."""
    return plt.cm.get_cmap(name, n)


def intersect(xs, ys):
    return max(xs[1], ys[1]) - min(xs[0], ys[0]) - (xs[1] - xs[0]) - (ys[1] - ys[0])


def intersect_mem(xs, ys):
    return intersect(xs, ys) < 0


def intersect_lvr(xs, ys):
    return intersect(xs, ys) <= 0


def intersect_allocs(alloc1, alloc2):
    ((begin1, end1), (offset1, size1)) = alloc1
    ((begin2, end2), (offset2, size2)) = alloc2
    interse = intersect_lvr((begin1, end1), (begin2, end2)) and intersect_mem(
        (offset1, offset1 + size1), (offset2, offset2 + size2)
    )

    if interse:
        print((begin1, end1), (begin2, end2))
        print((offset1, offset1 + size1), (offset2, offset2 + size2))

    return interse


def make_memory_map(
        allocations,
        title,
        *,
        save=True,
        fp_dir="/home/mlevental/dev_projects/pytorch_dev/memory_allocator/benchmarks/memory_maps",
        rescale=True,
):
    fig, ax, x_min, x_max, y_min, y_max = _make_memory_map(
        allocations, title, save=save, fp_dir=fp_dir, rescale=rescale
    )
    add_envelope_to_memory_map(
        fig,
        ax,
        x_min,
        x_max,
        y_min,
        y_max,
        allocations,
        title,
        shade=True,
        save=save,
        fp_dir=fp_dir,
        rescale=rescale,
    )
    plt.close(fig)


def _make_memory_map(
        allocations,
        title,
        *,
        save=True,
        fp_dir="/home/mlevental/dev_projects/pytorch_dev/memory_allocator/benchmarks/memory_maps",
        rescale=True,
):
    fig, ax = plt.subplots(figsize=(50, 10))

    x_min, x_max, y_min, y_max = float("inf"), 0, float("inf"), 0
    # Create a Rectangle patch
    cmap = get_cmap(len(allocations))
    for i, ((begin, end), (offset, size)) in enumerate(allocations):
        if rescale:
            x, y = begin, (offset / 2 ** 20)
            width, height = end - begin, (size / 2 ** 20)
        else:
            x, y = begin, offset
            width, height = end - begin, size

        x_min = min(x, x_min)
        y_min = min(y, y_min)
        x_max = max(x_max, x + width)
        y_max = max(y_max, y + height)

        if "mip" in title or "cp" in title:
            rect = patches.Rectangle(
                (x, y), width, height, linewidth=0.5, facecolor="green", edgecolor="black"
            )
        else:
            rect = patches.Rectangle(
                (x, y),
                width,
                height,
                linewidth=0.5,
                edgecolor="black",
                facecolor=cmap(i),
            )
        ax.add_patch(rect)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    ax.set_xlabel("time")
    ax.set_ylabel(f"memory ({'mb' if rescale else 'b'})")
    ax.set_title(title)

    fig.tight_layout()
    if save:
        assert fp_dir is not None
        fig.savefig(f"{fp_dir}/{title.replace(' ', '_')}.pdf")
        fig.savefig(f"{fp_dir}/{title.replace(' ', '_')}.svg")

    return fig, ax, x_min, x_max, y_min, y_max


def add_envelope_to_memory_map(
        fig,
        ax,
        x_min,
        _x_max,
        _y_min,
        _y_max,
        allocations,
        title,
        *,
        shade=True,
        save=True,
        fp_dir="/home/mlevental/dev_projects/pytorch_dev/memory_allocator/benchmarks/memory_maps",
        rescale=True,
):
    envelope_x = []
    envelope_y = []
    allocations.sort(key=lambda x: x[0][0])
    for i, ((begin, end), (offset, size)) in enumerate(allocations):
        if offset + size > max(
                [offse + siz for (_, (offse, siz)) in allocations[i + 1:]],
                default=float("inf"),
        ):
            envelope_x.append(end)
            if rescale:
                envelope_y.append((offset + size) / 2 ** 20)
            else:
                envelope_y.append(offset + size)

    if envelope_y:
        envelope_x.insert(0, x_min)
        envelope_y.insert(0, envelope_y[0])
    else:
        print("no envelope")
        return

    line = ax.plot(envelope_x, envelope_y, marker="o", ls="--", color="r")

    if save:
        assert fp_dir is not None
        fig.savefig(f"{fp_dir}/{title.replace(' ', '_')}_with_envelope.pdf")
        fig.savefig(f"{fp_dir}/{title.replace(' ', '_')}_with_envelope.svg")

    if not shade:
        return fig, ax, x_min, _x_max, _y_min, _y_max

    ax.lines.remove(line[0])
    for i, x in enumerate(envelope_x[1:], start=1):
        ax.vlines(x=x, ymin=0, ymax=envelope_y[i], colors="r", ls="--")

    ax.set_xticks(envelope_x[1:])
    ax.set_xticklabels(
        [f"{envelope_y[i + 1]:.3f} mb" for i in range(len(envelope_x[1:]))], rotation=90
    )
    ax.fill_between(envelope_x, envelope_y, step="pre", alpha=0.4, zorder=100)

    fig.tight_layout()

    if save:
        assert fp_dir is not None
        fig.savefig(f"{fp_dir}/{title.replace(' ', '_')}_with_shading.pdf")
        fig.savefig(f"{fp_dir}/{title.replace(' ', '_')}_with_shading.svg")

    return fig, ax, x_min, _x_max, _y_min, _y_max


def split_call_stacks(fp, title):
    # llvm-addr2line -e pytorch_memory_allocator -Ci 10000903e --dsym-hint=pytorch_memory_allocator.dSYM
    trace_json = json.load(open(fp))
    for t in trace_json["traceEvents"]:
        if "args" in t and "Call stack" in t["args"]:
            t["args"]["Call stack"] = list(reversed(t["args"]["Call stack"].split(";")))
            for i, line in enumerate(t["args"]["Call stack"]):
                if not line: continue
                if not "torch" in line and "c10" not in line: continue
                # (0x10cd3e03f,10000903e in pytorch_memory_allocator)
                fn, offset_addr = line.split("+")
                addr = offset_addr.split("(")[1].split(",")[1].split()[0]
                # obj_file = offset_addr.split("(")[1].split(",")[1].split()[2][:-1]
                cmd = f"dwarfdump /home/mlevental/dev_projects/pytorch_dev/build-static/pytorch_memory_allocator.dSYM/Contents/Resources/DWARF/pytorch_memory_allocator --lookup {int(addr, 16)} | tail -1"
                stream = os.popen(cmd)
                output = stream.read()
                t["args"]["Call stack"][i] = f"{fn} @ ({output})"

    json.dump(
        sorted(trace_json["traceEvents"], key=lambda x: x["ts"]),
        open(f"trace_{title}.json", "w"),
    )


def get_mip_allocations(fp):
    res = ast.literal_eval(open(fp).read())
    d = {}
    for model_name, allocs in res.items():
        d[model_name] = {f"%{a[0]}": a[1:] for a in allocs}
    return d


def get_profile_mip_allocations_scratch(strat_dict):
    d = {}
    for model_name, strats in strat_dict.items():
        _, tasks, _ = strats["mem_events_per_op"]
        offsets, obj_value = solve_cp(tasks.items())
        d[model_name] = [offsets, obj_value]
    return d


def get_static_plan_mip_allocations_scratch(strat_dict):
    d = {}
    for model_name, strats in strat_dict.items():
        tasks = [
            ((lvr[0], lvr[1] + 1), size)
            for lvr, (offset, size) in strats["greedy_by_size_with_smallest_gap"][
                "allocations"
            ].items()
        ]
        offsets, obj_value = solve_cp(tasks)
        d[model_name] = {
            "allocations": {(lvr[0], lvr[1] - 1): reg for lvr, reg in offsets},
            "live_ranges": strats["greedy_by_size_with_smallest_gap"]["live_ranges"],
        }
    return d


strat_colors = {
    "greedy_by_breadth": "red",
    "greedy_by_longest_and_size_with_first_gap": "blue",
    "greedy_by_longest_and_size_with_smallest_gap": "green",
    "greedy_by_size_with_first_gap": "orange",
    "greedy_by_size_with_smallest_gap": "pink",
    "mip": "purple",
    "naive": "black",
    "linear_scan": "gray",
    "eager": "teal",

    "profiled greedy_by_breadth": "red",
    "profiled greedy_by_longest_and_size_with_first_gap": "blue",
    "profiled greedy_by_longest_and_size_with_smallest_gap": "green",
    "profiled greedy_by_size_with_first_gap": "orange",
    "profiled greedy_by_size_with_smallest_gap": "pink",
    "profiled mip": "purple",
    "profiled naive": "black",
}


def plot_mem_usage(mem_usage, title, normalizer_name="mip", logy=False, last_one="linear_scan", ylabel="% max mem"):
    strategies = set()
    for model, strats in mem_usage.items():
        if "naive" in strats:
            strats.pop("naive")
        if "profiled naive" in strats:
            strats.pop("profiled naive")
        strategies = strategies.union(strats.keys())
    labels = list(mem_usage.keys())
    xs = (len(strategies) + 5) * np.arange(len(labels))  # the label locations
    label_to_x = dict(zip(labels, xs))
    width = 1  # the width of the bars

    fig, ax = plt.subplots(figsize=(12, 8))
    rectss = defaultdict(dict)
    for model, strat_results in mem_usage.items():
        normalizer = strat_results[normalizer_name]
        if last_one in strat_results:
            last_one_res = strat_results.pop(last_one)
            strat_results[last_one] = last_one_res
        for i, (strat, mem) in enumerate(strat_results.items()):
            mem /= normalizer
            x = label_to_x[model] - len(strategies) // 2 + i
            rectss[strat][model] = (x, mem)

    rects = {}
    if last_one in rectss:
        last_one_res = rectss.pop(last_one)
        rectss[last_one] = last_one_res
    for strat, models in rectss.items():
        print(strat, [f"{m[1]:.2f}" for m in models.values()])
        rects[strat] = ax.bar(
            [m[0] for m in models.values()],
            [m[1] for m in models.values()],
            width,
            color=strat_colors[strat],
            label=strat.replace("profiled ", "")
                .replace("static ", "")
                .replace("_", " ")
                .upper(),
        )

    # Add some text for labels, title and custom x-axis tick labels, etc.
    if logy:
        ax.set_yscale("log")
        # ax.set_ylim(1)
    # ax.set_ylabel(("log " if logy else "") + ylabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=75)
    ax.legend(bbox_to_anchor=(0.55, 1.1), loc="lower left")

    for strat in strategies:
        models = rectss[strat]
        labels = [f"{m[1]:.2f}" for m in models.values()]
        ax.bar_label(rects[strat], labels, padding=3, rotation=90)

    fig.tight_layout()

    fig.savefig(
        f"/home/mlevental/dev_projects/pytorch_dev/memory_allocator/benchmarks/{title.replace(' ', '_')}.pdf"
    )
    fig.savefig(
        f"/home/mlevental/dev_projects/pytorch_dev/memory_allocator/benchmarks/{title.replace(' ', '_')}.svg"
    )


def combine_strats(strat_results):
    outps = {}
    for model_name, strats in strat_results.items():
        outps[model_name] = set(
            strat_results[model_name]["greedy_by_size_with_smallest_gap"][
                "live_ranges"
            ].keys()
        )
        for strat_name, strat_dict in strats.items():
            outps[model_name] = outps[model_name].intersection(
                set(strat_results[model_name][strat_name]["live_ranges"].keys())
            )

    livrs = {}
    for model_name, strats in strat_results.items():
        livrs[model_name] = set(
            strat_results[model_name]["greedy_by_size_with_smallest_gap"][
                "live_ranges"
            ].items()
        )
        for strat_name, strat_dict in strats.items():
            livrs[model_name] = livrs[model_name].intersection(
                set(strat_results[model_name][strat_name]["live_ranges"].items())
            )
        livrs[model_name] = dict(livrs[model_name])
        assert max(Counter(livrs[model_name].values()).values()) == 1

    for model_name, strats in strat_results.items():
        for strat_name, strat_dict in strats.items():
            for outp in list(strat_dict["live_ranges"].keys()):
                if outp not in outps[model_name]:
                    lvr = strat_dict["live_ranges"][outp]
                    if lvr in strat_dict["allocations"]:
                        del strat_dict["allocations"][lvr]
                    del strat_dict["live_ranges"][outp]

    return strat_results


def get_mem_usage(strat_results):
    mem_usage = defaultdict(OrderedDict)
    num_allocs = defaultdict(OrderedDict)
    for model_name, strats in strat_results.items():
        for strat_name, strat_dict in sorted(strats.items(), key=lambda x: x[0]):
            allocations = strat_dict["allocations"].values()
            max_mem = 0
            for offset, size in allocations:
                max_mem = max(max_mem, offset + size)
            mem_usage[model_name][strat_name] = max_mem
            num_allocs[model_name][strat_name] = len(allocations)

    return dict(mem_usage), dict(num_allocs)


def get_strats_results_dict(fp):
    return ast.literal_eval(open(fp).read().lower())


def get_eager_mem_events(fp):
    js = json.load(open(fp))
    argss = []
    for j in js["traceEvents"]:
        if "[memory]" not in j["name"]:
            continue
        if "Call stack" in j["args"]:
            j["args"]["Call stack"] = j["args"]["Call stack"].split(";")
        j["args"]["ts"] = j["ts"]
        argss.append(j["args"])

    unique_tss = sorted(list(set([args["ts"] for args in argss])))
    ts_to_order = {ts: i for i, ts in enumerate(unique_tss)}
    max_ts = max(ts_to_order.values())

    addr_to_debug_name = {}
    debug_name_to_addrs = defaultdict(list)
    addr_live_range_mem = defaultdict(list)
    max_bytes = 0
    alloced_bytes = 0
    mem_events_new = {}

    for args in argss:
        addr = args["Addr"]
        bytes = args["Bytes"]
        # ts = ts_to_order[args["ts"]]
        ts = args["ts"]

        alloced_bytes += bytes
        max_bytes = max(max_bytes, alloced_bytes)

        addr_live_range_mem[addr].append(
            {
                "bytes": bytes,
                "ts": ts,
            }
        )
        if "NodeHeader" in args:
            debug_name = args["NodeHeader"].split(":")[0].strip().replace(".", "_")
            debug_name_to_addrs[debug_name].append(
                {
                    "ptr_addr": addr,
                    "bytes": bytes,
                    "ts": ts,
                }
            )
            addr_to_debug_name[addr] = debug_name
        else:
            assert bytes < 0
            assert addr in addr_to_debug_name
            debug_name_to_addrs[addr_to_debug_name[addr]].append(
                {
                    "ptr_addr": addr,
                    "bytes": bytes,
                    "ts": ts,
                }
            )

    debug_name_to_events = defaultdict(list)
    for addr, mems in addr_live_range_mem.items():
        for i in range(0, len(mems), 2):
            bytes, ts = mems[i]["bytes"], mems[i]["ts"]
            if i + 1 < len(mems):
                next_bytes, next_ts = mems[i + 1]["bytes"], mems[i + 1]["ts"]
                assert bytes == -next_bytes
                assert next_ts >= ts
                mem_events_new[ts, next_ts] = bytes
                debug_name_to_events[addr_to_debug_name[addr]].append(
                    ((ts, next_ts), bytes)
                )
            else:
                assert max_ts - ts < 20, f"{max_ts}, {ts}"

    return dict(debug_name_to_events), mem_events_new, max_bytes


def get_eager_op_lifetimes(fp):
    js = json.load(open(fp))

    repeats = defaultdict(lambda: 0)
    node_to_names = defaultdict(list)

    node_to_ts = defaultdict(list)
    for j in js["traceEvents"]:
        node_to_ts[j["name"] + f'{repeats[j["name"]]}'].append(j["ts"])
        repeats[j["name"]] += 1
        if "args" in j and "NodeHeader" in j["args"]:
            node_to_ts[j["args"]["NodeHeader"]].append(j["ts"])
            if j["name"] == "[memory]":
                node_to_names[j["args"]["NodeHeader"]].append(
                    (j["name"], j["args"]["Bytes"], j["args"]["Addr"], j["args"]["Call stack"].split(";")))
            else:
                node_to_names[j["args"]["NodeHeader"]].append(j["name"])
    node_to_lifetime = {}
    for node_header, tss in node_to_ts.items():
        node_to_lifetime[node_header] = (min(tss), max(tss))

    return node_to_lifetime, node_to_names


def find_smallest_gap(record, ordered_allocs):
    (begin_t, end_t), size_t = record
    best_gap = float("inf")
    best_offset = None
    prev_offset = 0

    for (begin_x, end_x), (offset_x, size_x) in ordered_allocs:
        if not intersect_lvr((begin_x, end_x), (begin_t, end_t)):
            continue

        # offset_x will be ahead of the previous block
        # while prev_offset will be just in front
        # this looks for small gap ahead of a block
        gap = offset_x - prev_offset
        if size_t <= gap < best_gap:
            best_gap = gap
            best_offset = prev_offset

        prev_offset = max(prev_offset, offset_x + size_x)

    if best_offset is None:
        best_offset = prev_offset
    return best_offset


def find_first_gap(record, ordered_allocs):
    (begin_t, end_t), size_t = record
    best_gap = float("inf")
    best_offset = None
    prev_offset = 0

    for (begin_x, end_x), (offset_x, size_x) in ordered_allocs:
        if not intersect_lvr((begin_x, end_x), (begin_t, end_t)):
            continue

        gap = offset_x - prev_offset
        if size_t <= gap < best_gap:
            best_offset = prev_offset
            break

        prev_offset = max(prev_offset, offset_x + size_x)

    if best_offset is None:
        best_offset = prev_offset
    return best_offset


def greedy_by_size_with_smallest_gap(ordered_records):
    ordered_records.sort(key=lambda x: -x[1])
    ordered_allocs = []
    inorder_of_decision_allocs = []
    total_consumption = 0
    for record in ordered_records:
        best_offset = find_smallest_gap(record, ordered_allocs)

        (begin_t, end_t), size_t = record
        total_consumption = max(total_consumption, best_offset + size_t)

        inorder_of_decision_allocs.append(((begin_t, end_t), (best_offset, size_t)))
        ordered_allocs.append(((begin_t, end_t), (best_offset, size_t)))
        ordered_allocs.sort(key=lambda x: x[1][0])

    return inorder_of_decision_allocs, total_consumption


def greedy_by_size_with_first_gap(ordered_records):
    ordered_records.sort(key=lambda x: -x[1])
    ordered_allocs = []
    inorder_of_decision_allocs = []
    total_consumption = 0
    for record in ordered_records:
        best_offset = find_first_gap(record, ordered_allocs)

        (begin_t, end_t), size_t = record
        total_consumption = max(total_consumption, best_offset + size_t)

        inorder_of_decision_allocs.append(((begin_t, end_t), (best_offset, size_t)))
        ordered_allocs.append(((begin_t, end_t), (best_offset, size_t)))
        ordered_allocs.sort(key=lambda x: x[1][0])

    return inorder_of_decision_allocs, total_consumption


def greedy_by_longest_and_size_with_first_gap(ordered_records):
    ordered_records.sort(key=lambda x: (x[0][1] - x[0][0]), reverse=True)
    ordered_allocs = []
    inorder_of_decision_allocs = []
    total_consumption = 0
    for record in ordered_records:
        best_offset = find_first_gap(record, ordered_allocs)

        (begin_t, end_t), size_t = record
        total_consumption = max(total_consumption, best_offset + size_t)

        ordered_allocs.append(((begin_t, end_t), (best_offset, size_t)))
        inorder_of_decision_allocs.append(((begin_t, end_t), (best_offset, size_t)))
        ordered_allocs.sort(key=lambda x: x[1][0])

    return inorder_of_decision_allocs, total_consumption


def greedy_by_longest_and_size_with_smallest_gap(ordered_records):
    ordered_records.sort(key=lambda x: (x[0][1] - x[0][0]), reverse=True)
    ordered_allocs = []
    inorder_of_decision_allocs = []
    total_consumption = 0
    for record in ordered_records:
        best_offset = find_smallest_gap(record, ordered_allocs)

        (begin_t, end_t), size_t = record
        total_consumption = max(total_consumption, best_offset + size_t)

        ordered_allocs.append(((begin_t, end_t), (best_offset, size_t)))
        inorder_of_decision_allocs.append(((begin_t, end_t), (best_offset, size_t)))
        ordered_allocs.sort(key=lambda x: x[1][0])

    return inorder_of_decision_allocs, total_consumption


def greedy_by_breadth(mem_events_per_op):
    mem_events_per_op_ls = list(mem_events_per_op.items())
    mem_events_per_op_ls.sort(key=lambda x: sum([y[1] for y in x[1]]), reverse=True)

    ordered_records = []
    for _, mem_events in mem_events_per_op_ls:
        mem_events.sort(key=lambda x: x[1], reverse=True)
        ordered_records.extend(mem_events)

    ordered_allocs = []
    inorder_of_decision_allocs = []
    total_consumption = 0
    for record in ordered_records:
        best_offset = find_smallest_gap(record, ordered_allocs)

        (begin_t, end_t), size_t = record
        total_consumption = max(total_consumption, best_offset + size_t)

        inorder_of_decision_allocs.append(((begin_t, end_t), (best_offset, size_t)))
        ordered_allocs.append(((begin_t, end_t), (best_offset, size_t)))
        ordered_allocs.sort(key=lambda x: x[1][0])

    return inorder_of_decision_allocs, total_consumption


def verify_allocation(allocations):
    for i, alloc1 in enumerate(allocations):
        for j, alloc2 in enumerate(allocations):
            if i == j:
                continue
            if intersect_allocs(alloc1, alloc2):
                return False
    return True


strat_fn_map = {
    # "greedy_by_breadth": greedy_by_breadth,
    "greedy_by_size_with_first_gap": greedy_by_size_with_first_gap,
    "greedy_by_size_with_smallest_gap": greedy_by_size_with_smallest_gap,
    "greedy_by_longest_and_size_with_first_gap": greedy_by_longest_and_size_with_first_gap,
    "greedy_by_longest_and_size_with_smallest_gap": greedy_by_longest_and_size_with_smallest_gap,
}


def plot_strat_static_results(strat_results):
    for model_name, strats in strat_results.items():
        reqs = []
        for ident, lvr in planned_strat_results[model_name]["naive"]["live_ranges"].items():
            if lvr in planned_strat_results[model_name]["naive"]["allocations"]:
                reqs.append((lvr, planned_strat_results[model_name]["naive"]["allocations"][lvr][1]))
        for strat_name, strat_dict in sorted(strats.items(), key=lambda x: x[0]):
            total_size = (
                    sum([sz for off, sz in strat_dict["allocations"].values()]) / 2 ** 20
            )
            if total_size < 1:
                continue
            print(
                model_name,
                f"total managed by {strat_name}",
                total_size,
            )
            print(
                f"{model_name}_{strat_name} valid allocation {verify_allocation(strat_dict['allocations'].items())}"
            )

            # if strat_name in strat_fn_map:
            #     # if strat_name == "greedy_by_breadth":
            #     #     debug_name_to_events = defaultdict(list)
            #     #     for addr, mems in addr_live_range_mem.items():
            #     #         for i in range(0, len(mems), 2):
            #     #             bytes, ts = mems[i]["bytes"], mems[i]["ts"]
            #     #             if i + 1 < len(mems):
            #     #                 next_bytes, next_ts = mems[i + 1]["bytes"], mems[i + 1]["ts"]
            #     #                 assert bytes == -next_bytes
            #     #                 assert next_ts >= ts
            #     #                 mem_events_new[ts, next_ts] = bytes
            #     #                 debug_name_to_events[addr_to_debug_name[addr]].append(
            #     #                     ((ts, next_ts), bytes)
            #     #                 )
            #     #             else:
            #     #                 assert max_ts - ts < 20, f"{max_ts}, {ts}"
            #     inorder_of_decision_allocs, total_consumption = strat_fn_map[strat_name](reqs)
            #     make_memory_map(inorder_of_decision_allocs, f"{model_name} static {strat_name}")
            # else:
            make_memory_map(
                list(strat_dict["allocations"].items()),
                f"{model_name} static {strat_name}",
            )
    print()
    mem_usage, _ = get_mem_usage(strat_results)
    # plot_mem_usage(mem_usage, "memory usage for strats and models for static plans")


def make_planned_strat_results(fp, do_mip=True):
    strat_results = get_strats_results_dict(fp)
    if do_mip:
        mip_res_regs = get_static_plan_mip_allocations_scratch(strat_results)
        for model_name, res in mip_res_regs.items():
            strat_results[model_name]["mip"] = res
    strat_results = combine_strats(strat_results)

    return strat_results


def make_profile_strat_results(do_mip=True):
    model_names = [
        "bert",
        "unet",
        "resnet18",
        "resnet34",
        "resnet50",
        "resnet101",
        "resnet152",
        "dcgan",
        "small_bert",
    ]
    strats_results = defaultdict(dict)
    for model_name in model_names:
        mem_events_per_op, mem_events, max_bytes = get_eager_mem_events(
            f"/home/mlevental/dev_projects/pytorch_dev/memory_allocator/benchmarks/traces/trace_cpp_{model_name}.json"
        )
        strats_results[model_name]["mem_events_per_op"] = (
            mem_events_per_op,
            mem_events,
            max_bytes,
        )

        (
            naive_allocations,
            total_size,
        ) = naive(mem_events)
        verify_allocation(naive_allocations)
        print(
            f"{model_name}_profile_naive valid allocation {total_size}/{max_bytes} {verify_allocation(naive_allocations)}"
        )
        strats_results[model_name]["profiled naive"] = (
            naive_allocations,
            total_size,
        )

        (
            greedy_by_size_with_smallest_gap_allocations,
            total_size,
        ) = greedy_by_size_with_smallest_gap(list(mem_events.items()))
        verify_allocation(greedy_by_size_with_smallest_gap_allocations)
        print(
            f"{model_name}_profile_greedy_by_size_with_smallest_gap valid allocation {total_size}/{max_bytes} {verify_allocation(greedy_by_size_with_smallest_gap_allocations)}"
        )
        strats_results[model_name]["profiled greedy_by_size_with_smallest_gap"] = (
            greedy_by_size_with_smallest_gap_allocations,
            total_size,
        )

        (
            greedy_by_size_first_gap_allocations,
            total_size,
        ) = greedy_by_size_with_first_gap(list(mem_events.items()))
        verify_allocation(greedy_by_size_first_gap_allocations)
        print(
            f"{model_name}_profile_greedy_by_size_with_first_gap valid allocation {total_size}/{max_bytes} {verify_allocation(greedy_by_size_first_gap_allocations)}"
        )
        strats_results[model_name]["profiled greedy_by_size_with_first_gap"] = (
            greedy_by_size_first_gap_allocations,
            total_size,
        )

        (
            greedy_by_longest_and_size_with_first_gap_allocations,
            total_size,
        ) = greedy_by_longest_and_size_with_first_gap(list(mem_events.items()))
        verify_allocation(greedy_by_longest_and_size_with_first_gap_allocations)
        print(
            f"{model_name}_profile_greedy_by_longest_and_size_with_first_gap valid allocation {total_size}/{max_bytes} {verify_allocation(greedy_by_longest_and_size_with_first_gap_allocations)}"
        )
        strats_results[model_name][
            "profiled greedy_by_longest_and_size_with_first_gap"
        ] = (
            greedy_by_longest_and_size_with_first_gap_allocations,
            total_size,
        )

        (
            greedy_by_longest_and_size_with_smallest_gap_allocations,
            total_size,
        ) = greedy_by_longest_and_size_with_smallest_gap(list(mem_events.items()))
        verify_allocation(greedy_by_longest_and_size_with_smallest_gap_allocations)
        print(
            f"{model_name}_greedy_by_longest_and_size_with_smallest_gap valid allocation {total_size}/{max_bytes} {verify_allocation(greedy_by_longest_and_size_with_first_gap_allocations)}"
        )
        strats_results[model_name][
            "profiled greedy_by_longest_and_size_with_smallest_gap"
        ] = (
            greedy_by_longest_and_size_with_smallest_gap_allocations,
            total_size,
        )

        greedy_by_breadth_allocations, total_size = greedy_by_breadth(mem_events_per_op)
        verify_allocation(greedy_by_breadth_allocations)
        print(
            f"{model_name}_profile_greedy_by_breadth valid allocation {total_size}/{max_bytes} {verify_allocation(greedy_by_breadth_allocations)}"
        )
        strats_results[model_name]["profiled greedy_by_breadth"] = (
            greedy_by_breadth_allocations,
            total_size,
        )

    if do_mip:
        mip_res_regs = get_profile_mip_allocations_scratch(strats_results)
        for model_name, res in mip_res_regs.items():
            strats_results[model_name]["profiled mip"] = res

    return dict(strats_results)


def miqp(tasks):
    import cvxpy as cp

    N = len(tasks)
    idents = []
    mems = []  # mems
    begins = []  # begins
    ends = []  # ends
    for ident, mem, begin, end in tasks:
        idents.append(ident)
        mems.append(mem)
        begins.append(begin)
        ends.append(end)
    max_mem = sum(mems)

    total_mem = cp.Variable(name="total_mem", integer=True)
    total_mem_pos_constraint = [0 <= total_mem]

    offsets = [
        cp.Variable(name=f"{begin, end}", integer=True)
        for (begin, end) in zip(begins, ends)
    ]
    pos_mem_constraints = [0 <= o for o in offsets]
    max_mem_constraints = [o <= max_mem for o in offsets]
    offset_mem_constraints = [
        offsets[i] + mems[i] <= total_mem for i in range(len(offsets))
    ]

    overlaps = []
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            if intersect_lvr([begins[i], ends[i]], [begins[j], ends[j]]):
                overlaps.append((i, j))
    overlaps = list(set(overlaps))

    order = {}
    for i, j in overlaps:
        order[i, j] = cp.Variable(name=f"z_{{{i},{j}}}", boolean=True)

    order_constraints = []
    for (i, j), z_ij in order.items():
        # if i < j then offsets[i] + mems[i] <= offsets[j]
        # but if j < i then offsets[i] + mems[i] <= offsets[j] + max_mem
        # otherwise the signs wouldn't be right
        order_constraints.append(offsets[i] + mems[i] <= offsets[j] + z_ij * max_mem)
        # similarly here
        order_constraints.append(
            offsets[j] + mems[j] <= offsets[i] + (1 - z_ij) * max_mem
        )

    # problem = cp.Problem(cp.Minimize(total_mem),
    #                      order_constraints + total_mem_pos_constraint + pos_mem_constraints + max_mem_constraints + offset_mem_constraints)

    problem = cp.Problem(
        cp.Minimize(cp.sum(cp.hstack(offsets))),
        order_constraints
        + total_mem_pos_constraint
        + pos_mem_constraints
        + max_mem_constraints
        + offset_mem_constraints,
    )
    problem.solve(solver=cp.SCIP, verbose=True)

    print("Status: ", problem.status)
    print("The optimal value is", problem.value)
    for i, offset in enumerate(offsets):
        print(f"{offset.name()}: ({offset.value}, {mems[i]})")
    # for (i, j), z_ij in order.items():
    #     print(i, j, z_ij.value)

    return [
               (ast.literal_eval(offset.name()), (offset.value.item(), mems[i]))
               for i, offset in enumerate(offsets)
           ], problem.value


def get_mem_usage_profile(strat_results):
    mem_usage = defaultdict(dict)
    for model_name, strats in strat_results.items():
        if "mem_events_per_op" in strats:
            strats.pop("mem_events_per_op")
        for strat_name, (allocations, total_size) in sorted(
                strats.items(), key=lambda x: x[0]
        ):
            max_mem = 0
            allocations.sort(key=lambda x: x[0])
            for _lvrs, (offset, size) in allocations:
                max_mem = max(max_mem, offset + size)
            mem_usage[model_name][strat_name] = max_mem
    return dict(mem_usage)


def naive(mem_events):
    allocations = []
    offset = 0
    size = 0
    for lvr, size in mem_events.items():
        allocations.append((lvr, (offset, size)))
        offset += size
    return allocations, offset + size


def plot_strat_results_profile(strat_results):
    for model_name, strats in strat_results.items():
        reqs = []
        for ident, lvr in planned_strat_results[model_name]["naive"]["live_ranges"].items():
            if lvr in planned_strat_results[model_name]["naive"]["allocations"]:
                reqs.append((lvr, planned_strat_results[model_name]["naive"]["allocations"][lvr][1]))

        strats.pop("mem_events_per_op")
        for strat_name, (allocations, total_size) in strats.items():
            print(
                model_name,
                f" total managed by {strat_name}",
                sum([sz for _lvr, (off, sz) in allocations]) / 2 ** 20,
            )
            print(
                f"{model_name}_{strat_name} valid allocation {verify_allocation(allocations)}"
            )

            if strat_name in strat_fn_map:
                inorder_of_decision_allocs, total_consumption = strat_fn_map[strat_name](reqs)
                make_memory_map(inorder_of_decision_allocs, f"{model_name} {strat_name}")
            else:
                make_memory_map(list(allocations), f"{model_name} {strat_name}")
    print()

    mem_usage = get_mem_usage_profile(strat_results)
    plot_mem_usage(mem_usage, "memory usage for strats and models for profiled plans", normalizer_name="profiled mip")


def total_usage(allocations):
    return sum([size for _lvr, (_off, size) in allocations])


def total_usage_mem_events(mem_events_per_op):
    return sum([size for _lvr, size in mem_events_per_op[1]])


def max_mem(allocations):
    allocations = list(sorted(allocations))
    max_mem = 0
    for _, (offset, size) in allocations:
        max_mem = max(max_mem, offset + size)
    return max_mem


def naive_compare_mem_usage(planned_results, profiled_results):
    max_mem_usage_planned, num_allocs_planned = get_mem_usage(planned_results)
    total_mem_usage_planned = defaultdict(dict)

    for model_name, strats in planned_results.items():
        for strat_name, strat_dict in strats.items():
            total_mem_usage_planned[model_name][strat_name] = total_usage(strat_dict["allocations"].items())

    total_mem_usage_profiled = defaultdict(dict)
    max_mem_usage_profiled = defaultdict(dict)
    num_allocs_profiled = defaultdict(dict)

    for model_name, strats in profiled_results.items():
        total_mem_usage_profiled[model_name]["eager"] = sum(strats["mem_events_per_op"][1].values())
        num_allocs_profiled[model_name]["eager"] = len(strats["mem_events_per_op"])
        max_mem_usage_profiled[model_name]["eager"] = strats["mem_events_per_op"][2]
        strats.pop("mem_events_per_op")
        for strat_name, strat_dict in strats.items():
            total_mem_usage_profiled[model_name][strat_name] = total_usage(strat_dict[0])
            num_allocs_profiled[model_name][strat_name] = len(strat_dict[0])
            max_mem_usage_profiled[model_name][strat_name] = max_mem(strat_dict[0])
            print()

    plot_mem_usage(total_mem_usage_profiled, "total mem usage", normalizer_name="eager", last_one="eager", ylabel="% total mem")
    plot_mem_usage(max_mem_usage_profiled, "max mem usage", normalizer_name="eager", last_one="eager", ylabel="% max mem")
    plot_mem_usage(num_allocs_profiled, "total num allocs", normalizer_name="eager", last_one="eager", ylabel="% num allocs")


def compare_mem_usage(planned_results, profiled_results, logy=False):
    fig, axs = plt.subplots(2, 1, figsize=(12, 12), sharex=True)

    model_names = []
    total_usages_static = []
    total_usages_profile = []
    num_allocs_static = []
    num_allocs_profile = []

    profiled_results = collections.OrderedDict(sorted(profiled_results.items()))
    for model_name, strats in profiled_results.items():
        model_names.append(model_name)
        total_usages_static.append(
            total_usage(
                planned_results[model_name]["greedy_by_size_with_smallest_gap"][
                    "allocations"
                ].items()
            )

        )
        total_usages_profile.append(
            total_usage(strats["profiled greedy_by_size_with_smallest_gap"][0])

        )

        num_allocs_static.append(
            len(
                planned_results[model_name]["greedy_by_size_with_smallest_gap"][
                    "allocations"
                ].items()
            )
        )
        num_allocs_profile.append(
            len(strats["profiled greedy_by_size_with_smallest_gap"][0])
        )
        print(
            model_name,
            total_usages_static[-1] / total_usages_profile[-1],
            num_allocs_static[-1] / num_allocs_profile[-1],
            sep=",",
        )

    total_usages_static = np.array(total_usages_static)
    total_usages_profile = np.array(total_usages_profile)
    num_allocs_static = np.array(num_allocs_static)
    num_allocs_profile = np.array(num_allocs_profile)

    width = 1
    xs = np.arange(len(model_names)) * len(model_names)
    bar1 = axs[0].bar(xs - 1, total_usages_static / total_usages_profile, width, label="static")
    bar2 = axs[0].bar(xs + 1, total_usages_profile / total_usages_profile, width, label="eager")
    if logy:
        axs[0].set_yscale("log")
    axs[0].set_ylabel(("log " if logy else "") + "% total mem")

    axs[0].bar_label(bar1, [f"{p:.2f}" for p in total_usages_static / total_usages_profile], padding=3, rotation=90)
    axs[0].bar_label(bar2, total_usages_profile / total_usages_profile, padding=3, rotation=90)

    bar1 = axs[1].bar(xs - 1, num_allocs_static / num_allocs_profile, width, label="static")
    bar2 = axs[1].bar(xs + 1, num_allocs_profile / num_allocs_profile, width, label="eager")
    axs[1].bar_label(bar1, [f"{p:.2f}" for p in num_allocs_static / num_allocs_profile], padding=3, rotation=90)
    axs[1].bar_label(bar2, num_allocs_profile / num_allocs_profile, padding=3, rotation=90)
    axs[1].set_ylabel("% total num mem events")
    if logy:
        axs[1].set_yscale("log")

    axs[0].set_title(("log " if logy else "") + "total mem tracked")
    axs[1].set_title("num mem events tracked")
    axs[1].set_xticks(xs)
    axs[1].set_xticklabels(model_names)
    axs[0].legend(loc="upper right", framealpha=0.5)
    # axs[0].legend(bbox_to_anchor=(1.05, 1))
    fig.tight_layout()

    fig.savefig(
        "/home/mlevental/dev_projects/pytorch_dev/memory_allocator/benchmarks/rel_mem_usage.pdf"
    )
    fig.savefig(
        "/home/mlevental/dev_projects/pytorch_dev/memory_allocator/benchmarks/rel_mem_usage.svg"
    )
    plt.close(fig)


def patholotical_case():
    print(intersect_lvr([1, 2], [2.5, 3]))
    print(intersect_lvr([1, 2], [2, 3]))
    print(intersect_mem([1, 2], [2, 3]))
    print(intersect_mem([1, 2], [1.5, 3]))

    allocs = [
        ((1630370401776956, 1630370401783399), (0, 2359296)),
        ((1630370401771493, 1630370401775434), (0, 1179648)),
        ((1630370401770769, 1630370401784731), (2359296, 492032)),
        ((1630370401776283, 1630370401783632), (2851328, 262144)),
        ((1630370401785532, 1630370401786715), (0, 262144)),
        ((1630370401771567, 1630370401776173), (1179648, 262144)),
        ((1630370401775632, 1630370401776742), (1441792, 262144)),
        ((1630370401784869, 1630370401786679), (262144, 262144)),
        ((1630370401777027, 1630370401786040), (3113472, 262144)),
        ((1630370401783966, 1630370401785401), (0, 262144)),
        ((1630370401783898, 1630370401784660), (2851328, 131072)),
        # ((1630370401775713, 1630370401776098), (1703936, 1024)),
        # ((1630370401775781, 1630370401776043), (1704960, 1024)),
        # ((1630370401784950, 1630370401785327), (524288, 1024)),
        # ((1630370401785018, 1630370401785274), (525312, 1024)),
        # ((1630370401785610, 1630370401785968), (524288, 1024)),
        # ((1630370401785677, 1630370401785915), (525312, 1024))
    ]
    make_memory_map(allocs, "corner_case_before")
    allocs, _ = greedy_by_size_with_smallest_gap(
        [(lvr, sz) for lvr, (off, sz) in allocs]
    )
    print(verify_allocation(allocs))
    make_memory_map(allocs, "corner_case_after")
    pprint(allocs)


def get_ls(i):
    linestyle_str = [
        ("solid", "solid"),  # Same as (0, ()) or '-'
        ("dotted", "dotted"),  # Same as (0, (1, 1)) or '.'
        ("dashed", "dashed"),  # Same as '--'
        ("dashdot", "dashdot"),  # Same as '-.'
        ("loosely dotted", (0, (1, 10))),
        ("dotted", (0, (1, 1))),
        ("densely dotted", (0, (1, 1))),
        ("loosely dashed", (0, (5, 10))),
        ("dashed", (0, (5, 5))),
        ("densely dashed", (0, (5, 1))),
        ("loosely dashdotted", (0, (3, 10, 1, 10))),
        ("dashdotted", (0, (3, 5, 1, 5))),
        ("densely dashdotted", (0, (3, 1, 1, 1))),
        ("dashdotdotted", (0, (3, 5, 1, 5, 1, 5))),
        ("loosely dashdotdotted", (0, (3, 10, 1, 10, 1, 10))),
        ("densely dashdotdotted", (0, (3, 1, 1, 1, 1, 1))),
    ]
    n_ls = len(linestyle_str)
    return linestyle_str[i % n_ls][1]


def small_bert_weird():
    node_to_lifetime = get_eager_op_lifetimes(
        "/home/mlevental/dev_projects/pytorch_dev/memory_allocator/benchmarks/traces/trace_cpp_resnet18.json"
    )
    mem_events_per_op, mem_events, max_bytes = get_eager_mem_events(
        f"/home/mlevental/dev_projects/pytorch_dev/memory_allocator/benchmarks/traces/trace_cpp_resnet18.json"
    )
    allocations, _total_size = naive(mem_events)
    fig, ax, x_min, x_max, y_min, y_max = _make_memory_map(
        allocations, "renset18 weird profile naive", save=False
    )

    cmap = get_cmap(len(allocations), "brg")
    xticks = []
    xticklabels = []
    for i, (node_header, (begin, end)) in enumerate(node_to_lifetime.items()):
        print(node_header, end - begin)
        ax.vlines(x=begin, ymin=0, ymax=y_max, colors=cmap(i), ls=get_ls(i))
        ax.vlines(x=end, ymin=0, ymax=y_max, colors=cmap(i), ls=get_ls(i))
        ax.hlines(y=y_max / 2, xmin=begin, xmax=end, colors=cmap(i), ls=get_ls(i))
        xticks.append((begin + end) / 2)
        xticklabels.append(node_header)

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, rotation=90)
    # ax.fill_between(envelope_x, envelope_y, step="pre", alpha=0.4, zorder=100)
    fig.set_size_inches(2000, 20)
    plt.subplots_adjust(bottom=0.5)
    # fig.tight_layout()
    fig.savefig(
        "/home/mlevental/dev_projects/pytorch_dev/memory_allocator/benchmarks/resnet18_weird_regions.pdf"
    )
    # plt.show()


if __name__ == "__main__":
    # small_bert_weird()
    # split_call_stacks("/home/mlevental/dev_projects/pytorch_dev/memory_allocator/benchmarks/traces/trace_cpp_resnet18.json", "resnet18_with_line_numbers")
    # small_bert_weird()
    # fp = "/home/mlevental/dev_projects/pytorch_dev/memory_allocator/benchmarks/cpp_strat_results.txt"
    # planned_strat_results = dict(make_planned_strat_results(fp, do_mip=True))
    # open(
    #     "/home/mlevental/dev_projects/pytorch_dev/memory_allocator/benchmarks/planned_strat_results.json",
    #     "w",
    # ).write(str(planned_strat_results))
    # planned_strat_results = get_strats_results_dict(
    #     "/home/mlevental/dev_projects/pytorch_dev/memory_allocator/benchmarks/planned_strat_results.json"
    # )
    # plot_strat_static_results(planned_strat_results)
    # mem_usage = get_mem_usage(planned_strat_results)
    # plot_mem_usage(mem_usage, "memory usage for strats and models for static plans")
    # res = get_static_plan_mip_allocations_scratch(planned_strat_results)

    # profiled_strat_results = make_profile_strat_results(do_mip=True)
    # open(
    #     "/home/mlevental/dev_projects/pytorch_dev/memory_allocator/benchmarks/profiled_strat_results.json",
    #     "w",
    # ).write(str(profiled_strat_results))
    # profiled_strat_results = get_strats_results_dict(
    #     "/home/mlevental/dev_projects/pytorch_dev/memory_allocator/benchmarks/profiled_strat_results.json"
    # )
    # plot_strat_results_profile(profiled_strat_results)

    # naive_res, _ = naive(profiled_strat_results["dcgan"]['mem_events_per_op'][1])
    # pprint(naive_res)
    # print(verify_allocation(naive_res))
    # fig, *_ = make_memory_map(naive_res, "naive", save=False)
    # fig.show()
    # res = solve_cp(profiled_strat_results["dcgan"]['mem_events_per_op'][1].items())
    # fig, *_ = make_memory_map(res[0], "dcgan_cp_sat", save=True, fp_dir="/Users/mlevental/dev_projects/pytorch_dev/memory_allocator/benchmarks/")
    # fig.show()
    # pprint(res[0])
    # print(verify_allocation(res[0]))
    # compare_mem_usage(planned_strat_results, profiled_strat_results)
    # naive_compare_mem_usage(planned_strat_results, profiled_strat_results)

    # # miqp(planned_strat_results)
    # # miqp(profiled_strat_results)
    #
    # # mem_events_per_op, mem_events, max_bytes = get_eager_mem_events(
    # #     f"/Users/mlevental/dev_projects/pytorch_dev/memory_allocator/benchmarks/traces/trace_cpp_resnet18.json"
    # # )

    fp = "/home/mlevental/dev_projects/pytorch_dev/memory_allocator/benchmarks/cpp_new_overlap_strats.txt"
    planned_strat_results = dict(make_planned_strat_results(fp, do_mip=True))
    for model_name, strats in planned_strat_results.items():
        for strat, strat_dict in strats.items():
            print(model_name, strat, verify_allocation(strat_dict["allocations"].items()))