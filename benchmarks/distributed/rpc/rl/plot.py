import matplotlib
import matplotlib.pyplot as plt
import numpy as np


labels = list(range(1, 11))
batch_execution_time = [94.32738304138184, 94.69485807418823, 95.18068504333496, 95.85501790046692,
                        96.67237305641174, 97.23094129562378, 98.4750702381134, 98.37114596366882, 98.32534098625183, 98.53375101089478]
non_batch_execution_time = [93.64854669570923, 96.23795795440674, 98.69362783432007, 102.37100005149841,
                            103.5243968963623, 106.74777126312256, 112.08952116966248, 112.05054187774658, 114.44980812072754, 115.52028274536133]

batch_execution_time = [round(b, 1) for b in batch_execution_time]
non_batch_execution_time = [round(b, 1) for b in non_batch_execution_time]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, batch_execution_time, width, label='Batch')
rects2 = ax.bar(x + width/2, non_batch_execution_time,
                width, label='Non-Batch')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('10-Episodes Execution Time (Seconds)')
ax.set_xlabel('Number of Observers')
ax.set_title('RPC Benchmarks')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()


# def autolabel(rects):
#     """Attach a text label above each bar in *rects*, displaying its height."""
#     for rect in rects:
#         height = rect.get_height()
#         ax.annotate('{}'.format(height),
#                     xy=(rect.get_x() + rect.get_width() / 2, height),
#                     xytext=(0, 3),  # 3 points vertical offset
#                     textcoords="offset points",
#                     ha='center', va='bottom')


# autolabel(rects1)
# autolabel(rects2)

fig.tight_layout()

plt.grid()
plt.show()
