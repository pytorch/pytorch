from .CPUMetric import CPUMetric
from .CUDAMetric import CUDAMetric


class MetricsLogger:
    def __init__(self, rank=None):
        self.rank = rank
        self.metrics = {}

    def record_start(self, type, key, name, cuda):
        if type in self.metrics and key in self.metrics[type]:
            raise RuntimeError(f"metric_type={type} with key={key} already exists")
        if cuda:
            if self.rank is None:
                raise RuntimeError("rank is required for cuda")
            metric = CUDAMetric(self.rank, name)
        else:
            metric = CPUMetric(name)
        if type not in self.metrics:
            self.metrics[type] = {}
        self.metrics[type][key] = metric
        metric.record_start()

    def record_end(self, type, key):
        if type not in self.metrics or key not in self.metrics[type]:
            raise RuntimeError(f"metric_type={type} with key={key} not found")
        if self.metrics[type][key].get_end() is not None:
            raise RuntimeError(
                f"end for metric_type={type} with key={key} already exists"
            )
        self.metrics[type][key].record_end()

    def clear_metrics(self):
        self.metrics.clear()

    def get_metrics(self):
        return self.metrics

    def get_processed_metrics(self):
        r"""
        A method that processes the metrics recorded during the benchmark.

        Returns::
            It returns a dictionary containing keys as the metrics
                and values list of elapsed times.

        Examples::

            >>> instance = MetricsLogger(rank)
            >>> instance.cuda_record_start("forward_metric_type", "1", "forward_pass")
            >>> instance.cuda_record_end("forward_metric_type", "1")
            >>> instance.cuda_record_start("forward_metric_type", "2", "forward_pass")
            >>> instance.cuda_record_end("forward_metric_type", "2")
            >>> print(instance.metrics)
            {
                "forward_metric_type": {
                    "1": metric1,
                    "2": metric2
                }
            }

            >>> print(instance.get_processed_metrics())
            {
                "forward_metric_type,forward_pass" : [.0429, .0888]
            }
        """
        processed_metrics = {}
        for metric_type in self.metrics.keys():
            for metric_key in self.metrics[metric_type].keys():
                metric = self.metrics[metric_type][metric_key]
                if isinstance(metric, CUDAMetric):
                    metric.synchronize()
                metric_name = metric.get_name()
                elapsed_time = metric.elapsed_time()
                processed_metric_name = f"{metric_type},{metric_name}"
                if processed_metric_name not in processed_metrics:
                    processed_metrics[processed_metric_name] = []
                processed_metrics[processed_metric_name].append(elapsed_time)
        return processed_metrics
