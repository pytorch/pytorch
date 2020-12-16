import json
import matplotlib.pyplot as plt
import numpy as np

with open('report.json') as report:
        report = json.load(report)

x_axis_name = report['x_axis_name']
bar_width = 0.35  # the bar_width of the bars

benchmark_results = report['benchmark_results']
x_axis_labels = [benchmark_run[x_axis_name] for benchmark_run in benchmark_results]
label_location = np.arange(len(x_axis_labels))

for benchmark_metric in ['agent throughput', 'observer throughput', 'agent latency', 'observer latency']:
    fig, ax = plt.subplots()
    p50s = []
    p95s = []
    for i in range(len(x_axis_labels)):
        p50s.append(benchmark_results[i][benchmark_metric]['50'])
        p95s.append(benchmark_results[i][benchmark_metric]['95'])

    y1 = ax.bar(label_location - bar_width/2, p50s, bar_width, label='p50')
    y2 = ax.bar(label_location + bar_width/2, p95s, bar_width, label='p95')
    ax.set_ylabel(benchmark_metric)
    ax.set_xlabel(x_axis_name)
    ax.set_title('RPC Benchmarks')
    ax.set_xticks(label_location)
    ax.set_xticklabels(x_axis_labels)
    ax.legend()
    fig.tight_layout()
plt.grid()
plt.show()