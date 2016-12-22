from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import json
import re
import sys

display_levels = ["network", "worker", "operator", "kernel"]


def stop_display(limit, curr):
    return display_levels.index(limit) <= display_levels.index(curr)


def build_trace_dict(f):
    """Creates a python dictionary that has trace ids as keys and the
    corresponding trace objects as values.

    Input: python file object that points to a file with traces, written by
    htrace-c's local file span receiver.
    The exact format shouldn't concern you if you're using htrace-c correctly.
    https://github.com/apache/incubator-htrace/blob/master/htrace-c.

    Returns: a tuple (trace_dic, root_list), where trace_dic is a dictionary
    containing all traces parsed from the input file object, and root_list is a
    list of traces from trace_dic which have no parents.
    Each value in trace_dic is in the form of another dictionary with the
    folowing keys:
        "begin"   : timestamp of trace start time, microseconds
        "end"     : timestamp of trace end time, microseconds
        "desc"    : description of trace
        "parent"  : trace id of parent trace
        "children": dictionary of child traces, in the same format as trace_dic
    """
    trace_dic = {}
    root_list = []
    for line in f:
        h = json.loads(line)
        entry = {"begin": h["b"], "end": h["e"], "desc": h["d"]}
        if "p" not in h or len(h["p"]) == 0:
            root_list.append(entry)
        else:
            entry["parent"] = h["p"][0]
        trace_dic[h["a"]] = entry

    for k, v in trace_dic.items():
        if "parent" not in v:
            continue
        parent = trace_dic[v["parent"]]
        if "children" not in parent:
            parent["children"] = {}
        parent["children"][k] = v

    return trace_dic, root_list


def generate_chrome_trace(root_list, display):
    """Takes trace objects created by build_trace_dict() and generates a list of
    python dictionaries that can be written to a file in json format, which in
    turn can be given to Chrome tracing (chrome://tracing).

    Input: refer to root_list in build_trace_dict()'s return value.

    Output: list of dictionaries that can be directly written to a json file by
    json.dumps().
    The dictionary format follows the JSON array format of Chrome tracing.
    Complete events ("ph": "X") are used to express most traces; such events
    will appear as horizontal blocks with lengths equal to the trace duration.
    Instant events ("ph": "i") are used for traces with many occurrencs which
    may make the trace graph unreadable; such events are shown as thin lines.
    """
    ct = []
    for root_idx, root in enumerate(root_list):
        # network-level spans
        ct.append({
            "name": root["desc"],
            "ph": "X",
            "ts": root["begin"],
            "dur": root["end"] - root["begin"],
            "pid": root_idx,
            "tid": root_idx
        })

        for _, v in root["children"].items():
            # run-scopes and worker-scopes
            c = {
                "name": v["desc"],
                "ph": "X",
                "ts": v["begin"],
                "dur": v["end"] - v["begin"],
                "pid": root_idx
            }

            if "run-scope" in v["desc"]:
                c["tid"] = root_idx
                ct.append(c)
            else:
                if stop_display(display, "network"):
                    continue

                m = re.search("(?<=worker-scope-)\d+", v["desc"])
                wid = m.group(0)
                c["tid"] = wid
                ct.append(c)

                if stop_display(display, "worker") or "children" not in v:
                    continue
                for k_op, v_op in v["children"].items():
                    # operator scopes
                    ct.append({
                        "name": v_op["desc"],
                        "ph": "X",
                        "ts": v_op["begin"],
                        "dur": v_op["end"] - v_op["begin"],
                        "pid": root_idx,
                        "tid": wid
                    })

                    if stop_display(display, "operator") or "children" not in v_op:
                        continue
                    for idx, (k_gpu_op, v_gpu_op) in \
                            enumerate(sorted(v_op["children"].items(),
                                             key=lambda e: e[1]["begin"])):
                        # kernel scopes
                        if idx == 0:
                            ct.append({
                                "name": v_op["desc"] + "-GPU",
                                "ph": "X",
                                "ts": v_gpu_op["begin"],
                                "dur": v_gpu_op["end"] - v_gpu_op["begin"],
                                "pid": root_idx,
                                "tid": wid,
                                "args": {
                                    "desc": "NEW OPERATOR"
                                }
                            })

                        ct.append({
                            "name": v_op["desc"] + "-GPU",
                            "ph": "i",
                            "ts": v_gpu_op["begin"],
                            "pid": root_idx,
                            "tid": wid,
                            "args": {
                                "desc": v_gpu_op["desc"]
                            }
                        })

    return ct


def get_argument_parser():
    parser = argparse.ArgumentParser(
        description="Format conversion from HTrace to Chrome tracing.")
    parser.add_argument("htrace_log", type=str, help="input htrace span log file")
    parser.add_argument("--display",
                        type=str, choices=display_levels, default="operator",
                        help="deepest level of spans to display (default: operator)")
    return parser


def main():
    args = get_argument_parser().parse_args()
    with open(args.htrace_log, "r") as f:
        trace_dic, root_list = build_trace_dict(f)

    ct = generate_chrome_trace(root_list, args.display)
    print("Writing chrome json file to %s.json" % sys.argv[1])
    print("Now import %s.json in chrome://tracing" % sys.argv[1])
    with open(sys.argv[1] + ".json", "w") as f:
        f.write(json.dumps(ct))


if __name__ == '__main__':
    main()
