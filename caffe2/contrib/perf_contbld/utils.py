## @package utils
# Module caffe2.contrib.perf_contbld.utils
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


from caffe2.proto import prof_dag_pb2

class OperatorStatsContainer():
    '''
    This class works as a wrapper to print ProfDAGNet statistics
    '''
    def __init__(self, stats_proto):
        self.op_stats = prof_dag_pb2.ProfDAGProtos()
        self.op_stats.ParseFromString(stats_proto)

    def Print(self):
        print("Time per operator type:")
        for stat in self.op_stats.stats:
            print("{:12.6f} ms/iter [{:10.6f} ms/iter  ]\t{}".format(
                stat.mean, stat.stddev, stat.name
            ))
