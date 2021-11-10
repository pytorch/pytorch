import pandas

df = pandas.read_csv("perf.csv")

ops = pandas.unique(df["operator"])
nops = len(ops)
pivot_op_shape = df.pivot_table(
    values="time", index=["operator", "shape"], columns=["fuser"]
)
pivot_speedups = (pivot_op_shape.T / pivot_op_shape["eager"]).T

import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (20, 100)
fig, axs = plt.subplots(nops)
plt.subplots_adjust(hspace=0.5)
for idx, op in enumerate(ops):
    op_speedups = pivot_speedups.T[op].T
    op_speedups.plot(ax=axs[idx], kind="bar", ylim=(0, 2), rot=45)
    axs[idx].set_title(op)
    axs[idx].set_xlabel("")
plt.savefig("perf.png")
