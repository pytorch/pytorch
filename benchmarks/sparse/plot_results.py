import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import argparse

def plot_results(filepath): 
    def rel_plot(df):
        ax = sns.relplot(
            data=df, x="sparsity", y="time",
            col="device", hue="method", style="method",
            kind="line"
        )
        ax.set_axis_labels('sparsity', 'time (ms)')

    def line_plot(df, method, device):
        df=df[df.method.str.contains(method)]
        df=df[df.device.str.contains(device)]
        ax = sns.lineplot(df.sparsity, df.time, label=method, err_style="bars", ci=20)
        ax.set(title=f"sparse matmul({device})")
        ax.set_axis_labels('sparsity', 'time (ms)')

    df = pd.read_pickle(filepath)
    rel_plot(df)
    plt.show()

path = Path()
parser = argparse.ArgumentParser(description='Sparse Matmul Bench')

parser.add_argument('--path', '-p',action='store', dest='rn50_path',
                    help='rn50 dataset path', default=path.cwd()/'rn50/')
parser.add_argument('--dataset', '-d',action='store', dest='dataset',
                    help='rn50 dataset path', default='random_pruning')
parser.add_argument('--input', '-i',action='store', dest='input',
                    help='dataframe input path', default='/tmp/matmul_bench.pkl')
results = parser.parse_args()
print ('input     =', results.input)
 
plot_results(results.input)
