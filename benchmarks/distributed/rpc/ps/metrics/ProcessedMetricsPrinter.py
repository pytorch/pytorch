import statistics

import pandas as pd
from tabulate import tabulate


class ProcessedMetricsPrinter:

    def combine_processed_metrics(self, processed_metrics_list):
        processed_metric_totals = {}
        for processed_metrics in processed_metrics_list:
            for metric_name, values in processed_metrics.items():
                if metric_name not in processed_metric_totals:
                    processed_metric_totals[metric_name] = []
                processed_metric_totals[metric_name] += values
        return processed_metric_totals

    def print_metrics(self, name, processed_metrics):
        print("metrics for {}".format(name))
        data_frame = self.get_data_frame(processed_metrics)
        print(tabulate(data_frame, showindex=False, headers=data_frame.columns, tablefmt="grid"))

    def get_data_frame(self, processed_metrics):
        df = pd.DataFrame(
            columns=['name', 'min', 'max', 'mean', 'variance', 'stdev']
        )
        for metric_name in sorted(processed_metrics.keys()):
            values = processed_metrics[metric_name]
            row = {
                "name": metric_name,
                "min": min(values),
                "max": max(values),
                "mean": statistics.mean(values),
                "variance": statistics.variance(values),
                "stdev": statistics.stdev(values)
            }
            df = df.append(row, ignore_index=True)
        return df

    def save_to_file(self, data_frame, file_name):
        file_name = "data_frames/{}.csv".format(
            file_name
        )
        data_frame.to_csv(file_name, encoding='utf-8', index=False)
