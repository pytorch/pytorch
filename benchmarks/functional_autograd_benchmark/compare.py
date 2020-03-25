import argparse
from collections import defaultdict

from utils import get_str, read_str

def main():
    parser = argparse.ArgumentParser("Main script to compare results from the benchmarks")
    parser.add_argument("--before", type=str, default="before.txt", help="Text file containing the times to use as base")
    parser.add_argument("--after", type=str, default="after.txt", help="Text file containing the times to use as new version")
    parser.add_argument("--output", type=str, default="", help="Text file where to write the output")
    args = parser.parse_args()

    with open(args.before, "r") as f:
        content = f.read()
    res1 = read_str(content)

    with open(args.after, "r") as f:
        content = f.read()
    res2 = read_str(content)

    diff = defaultdict(defaultdict)
    for model in res1:
        for task in res1[model]:
            mean1, var1 = res1[model][task]
            if task not in res2[model]:
                diff[model][task] = (-1, mean1, var1, -1, -1)
            else:
                mean2, var2 = res2[model][task]
                diff[model][task] = (mean1 / mean2, mean1, var1, mean2, var2)
    for model in res2:
        for task in res2[model]:
            if task not in res1[model]:
                mean2, var2 = res2[model][task]
                diff[model][task] = (-1, -1, -1, mean2, var2)

    header = ("model", "task", "speedup", "mean (before)", "var (before)", "mean (after)", "var (after)")
    out = get_str(diff, header=header)

    print(out)
    if args.output:
        with open(args.output, "w") as f:
            f.write(out)

if __name__ == "__main__":
    main()
