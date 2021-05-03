from .CPUMetric import CPUMetric
from .CUDAMetric import CUDAMetric


class MetricsLogger:

    CUDA = "cuda"
    CPU = "cpu"

    def __init__(self, rank, metric_class, overwrite_metrics=False):
        self.rank = rank
        self.metric_class = metric_class.lower()
        self.overwrite_metrics = overwrite_metrics
        self.metrics = {}

    def add_metric(self, metric_type, key, metric_name):
        if metric_type in self.metrics and key in self.metrics[metric_type] and not self.overwrite_metrics:
            raise Exception("metric_type={} with key={} already exists".format(metric_type, key))

        if metric_type not in self.metrics:
            self.metrics[metric_type] = {}

        if self.metric_class == self.CPU:
            self.metrics[metric_type][key] = CPUMetric(metric_name)
        elif self.metric_class == self.CUDA:
            self.metrics[metric_type][key] = CUDAMetric(metric_name)

        self.metrics[metric_type][key].record_start(self.rank)

    def add_metric_end(self, metric_type, key):

        if metric_type not in self.metrics or key not in self.metrics[metric_type]:
            raise Exception("metric_type={} with key={} not found".format(metric_type, key))

        if self.metrics[metric_type][key].get_end() is not None and not self.overwrite_metrics:
            raise Exception("end for metric_type={} with key={} already exists".format(metric_type, key))

        self.metrics[metric_type][key].record_end(self.rank)

    def clear_metrics(self):
        self.metrics.clear()

    def get_metrics(self):
        return self.metrics

    r"""Method that processes the Metrics. 
    It returns a dictionary containing keys 
    that are the metric_type and metric_name combined. 
    Each key has a list of elapsed times.

     Args:
        self: the instance of the class

     Example:

        metric1 = CUDAMetric("forward_pass")
        metric1.record_start(rank)
        metric1.record_end(rank)

        metric2 = CUDAMetric("forward_pass")
        metric2.record_start(rank)
        metric2.record_end(rank)

        self.metrics == { 
            "forward_metric_type": {
                "1": metric1,
                "2": metric2
            }
        }

        processed_metrics == {
            "forward_metric_type_forward_pass" : [.0429, .0888]
        }
    """

    def get_processed_metrics(self):
        processed_metrics = {}
        for metric_type in self.metrics.keys():
            for metric_key in self.metrics[metric_type].keys():
                metric = self.metrics[metric_type][metric_key]
                metric_name = metric.get_name()
                elapsed_time = metric.elapsed_time(self.rank)
                processed_metric_name = "{},{}".format(metric_type, metric_name)
                if processed_metric_name not in processed_metrics:
                    processed_metrics[processed_metric_name] = []
                processed_metrics[processed_metric_name].append(elapsed_time)
        return processed_metrics
