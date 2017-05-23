## @package utils
# Module caffe2.contrib.perf_contbld.utils
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import getpass
import time
from collections import defaultdict
import numpy as np

from caffe2.proto import prof_dag_pb2
from scubadata import Sample, ScubaData
from rfe import client as rfe_client
from RockfortExpress import RockfortExpress as rfe
from libfb.employee import unixname_to_uid


class OperatorStatsContainer():
    '''
    This class works as a wrapper to log ProfDAGNet statistics to Scuba
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

    def _scuba_query(self, sql):
        user_name = getpass.getuser()
        user_id = unixname_to_uid(user_name)
        query = rfe.QueryCommon(
            user_name=user_name,
            user_id=user_id,
        )
        return rfe_client.getClient().querySQL(query, sql)

    def _query(self, model, num_days):
        '''
        Given a model, returns the op stats
        '''
        cur_unix_epoch = int(time.time())

        sql = """ SELECT operator, op_mean, op_stddev
              FROM caffe2_op_runs
              WHERE model_name = \'{}\' and time >= {}
              """.format(model, cur_unix_epoch - num_days * 86400)
        result = self._scuba_query(sql)

        if not result.headers:
            return

        headers = result.headers
        op_idx = headers.index('operator')
        mean_idx = headers.index('op_mean')

        # dict (key, (value1, value2,...))
        # key: "operator" value: "op_mean"
        d = defaultdict(list)
        for row in result.value:
            d[row[op_idx]].append(float(row[mean_idx]))
        return d

    def ReadOpRuns(self, model, num_days):
        print("Reading op stats for model {}".format(model))
        return self._query(model, num_days)

    def WriteOpRuns(self, model):
        print("Logging to scuba for model {}".format(model))
        scuba = ScubaData("caffe2_op_runs")

        sample = Sample()
        sample.add_normal("model_name", model)

        for stat in self.op_stats.stats:
            sample.add_normal("operator", stat.name)
            sample.add_double("op_mean", stat.mean)
            sample.add_double("op_stddev", stat.stddev)
            scuba.add_sample(sample)

    def CheckRegression(
        self, model, num_days, min_exec_time, std_coefficient, mean_coefficient
    ):
        print("Regression check")
        op_runs = self.ReadOpRuns(model, num_days) or defaultdict(list)
        regression = False
        op_list = {}

        # Iterate over current run's operator timing
        for stat in self.op_stats.stats:
            times = op_runs[stat.name]
            print("{} execution times: {}".format(stat.name, times))

            mean = np.mean(times)
            std = np.std(times)

            if not times or stat.mean < min_exec_time:
                continue

            if stat.mean > (std_coefficient * std +
                            mean) and stat.mean > (mean_coefficient * mean):
                regression = True
                op_list[stat.name] = str((stat.mean - mean) * 100 / mean) + "%"
                print(
                    "\tregression for {}: current runtime {} ms".
                    format(stat.name, stat.mean)
                )

        if not regression:
            # Write the operator execution times to caffe2_op_runs table
            self.WriteOpRuns(model)
        else:
            raise Exception(op_list)
