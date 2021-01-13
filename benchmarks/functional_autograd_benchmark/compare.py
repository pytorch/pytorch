import argparse
from collections import defaultdict

from utils import to_markdown_table, from_markdown_table

def main():
    parser = argparse.ArgumentParser("Main script to compare results from the benchmarks")
    parser.add_argument("--before", type=str, default="before.txt", help="Text file containing the times to use as base")
    parser.add_argument("--after", type=str, default="after.txt", help="Text file containing the times to use as new version")
    parser.add_argument("--output", type=str, default="", help="Text file where to write the output")
    args = parser.parse_args()

    with open(args.before, "r") as f:
        content = f.read()
    res_before = from_markdown_table(content)

    with open(args.after, "r") as f:
        content = f.read()
    res_after = from_markdown_table(content)

    diff = defaultdict(defaultdict)
    for model in res_before:
        for task in res_before[model]:
            mean_before, var_before = res_before[model][task]
            if task not in res_after[model]:
                diff[model][task] = (None, mean_before, var_before, None, None)
            else:
                mean_after, var_after = res_after[model][task]
                diff[model][task] = (mean_before / mean_after, mean_before, var_before, mean_after, var_after)
    for model in res_after:
        for task in res_after[model]:
            if task not in res_before[model]:
                mean_after, var_after = res_after[model][task]
                diff[model][task] = (None, None, None, mean_after, var_after)

    header = ("model", "task", "speedup", "mean (before)", "var (before)", "mean (after)", "var (after)")
    out = to_markdown_table(diff, header=header)

    print(out)
    if args.output:
        with open(args.output, "w") as f:
            f.write(out)

if __name__ == "__main__":
    main()
