import torch

import csv
import json
import sys

from dataclasses import dataclass
from typing import List, Dict

class ComparisonResult():
    graph: str
    output_idx: int
    rel_diff: float
    abs_diff: float
    unfused_dtype: str
    fused_dtype: str
    ops_count: float

    def asdict(self):
        return {
            "graph": self.graph,
            "output_idx": self.output_idx,
            "rel_diff": self.rel_diff,
            "unfused_dtype": self.unfused_dtype,
            "fused_dtype": self.fused_dtype,
            "ops_count": self.ops_count,
        }

def count_nonconstant_ops(graph):
    count = 0
    for n in graph.nodes():
        if n.kind() != 'prim::Constant':
            count += 1
    return count

class CompareLogger():
    def __init__(self, print_all=True):
        self.print_all = print_all
        self.results = []

    def callback(self, fused_outputs, unfused_outputs, graph):
        for i in range(len(fused_outputs)):
            fused = fused_outputs[i]
            unfused = unfused_outputs[i]
            result = ComparisonResult()
            result.graph = str(graph)
            result.output_idx = i
            result.ops_count = count_nonconstant_ops(graph)
            result.abs_diff = torch.max(torch.abs(fused - unfused)).item()
            matches = (fused == unfused)
            rel_diff_tensor = torch.abs(fused - unfused) / torch.abs(unfused)
            rel_diff_tensor[matches] = 0
            result.rel_diff = torch.max(rel_diff_tensor).item()

            result.unfused_dtype = str(unfused.dtype)
            result.fused_dtype = str(fused.dtype)
            self.results.append(result)

    def dedup_graphs(self, graphs: List[str]) -> Dict[str, int]:
        counter = 1
        result = {}
        for g in graphs:
            if g not in result:
                result[g] = counter
                counter += 1

        return result

    def dump_without_graphs(self, dump_to=None, export_format='csv'):
        if dump_to is None:
            dump_to = sys.stdout
        graph_idx = self.dedup_graphs([x.graph for x in self.results])
        data = []
        for result in self.results:
            data_dict = result.asdict()
            data_dict["graph_idx"] = graph_idx[data_dict["graph"]]
            del data_dict["graph"]
            data.append(data_dict)
        if export_format == 'csv':
            cols = ["graph_idx", "output_idx", "rel_diff", "abs_diff", "unfused_dtype", "fused_dtype", "ops_count"]
            writer = csv.DictWriter(dump_to, fieldnames=cols)
            writer.writeheader()
            for d in data:
                writer.writerow(d)
        elif export_format == 'json':
            json.dump(data, dump_to, indent=4)

    def dump_graphs(self, dump_to=None):
        if dump_to is None:
            dump_to = sys.stdout

        graph_idx = self.dedup_graphs([x.graph for x in self.results])
        reverse = {v: k for k, v in graph_idx.items()}

        json.dump(reverse, dump_to, indent=4)
