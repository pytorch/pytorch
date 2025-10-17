results = []
for r in open("log_bert_pytorch_plain.out"):
    r = r.strip()
    if not r.startswith("HINT_SIZE"):
        continue
    hint_size = r.split("HINT_SIZE (")[1].split(") RUNTIME_SIZE")[0]
    runtime_size = r.split("RUNTIME_SIZE (")[1].split(")")[0]

    hint_bs, hint_sl = hint_size.split(", ")
    runtime_bs, runtime_sl = runtime_size.split(", ")

    perf = r.split(" ")[-1].split("x")[0]
    results.append(
        {
            "hint_bs": int(hint_bs),
            "hint_sl": int(hint_sl),
            "runtime_bs": int(runtime_bs),
            "runtime_sl": int(runtime_sl),
            "perf": float(perf),
        }
    )

import pandas as pd


df = pd.DataFrame(results)
df.to_csv("bert_pytorch_perf_plain.csv", index=False)
df = df.sort_values(
    by=["runtime_bs", "runtime_sl", "perf"], ascending=False, ignore_index=True
)
with pd.option_context(
    "display.max_rows", None, "display.max_columns", None
):  # more options can be specified also
    print(df)
