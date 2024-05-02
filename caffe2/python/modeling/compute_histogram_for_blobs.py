




from caffe2.python import core, schema
from caffe2.python.modeling.net_modifier import NetModifier

import numpy as np


class ComputeHistogramForBlobs(NetModifier):
    """
    This class modifies the net passed in by adding ops to compute histogram for
    certain blobs.

    Args:
        blobs: list of blobs to compute histogram for
        logging_frequency: frequency for printing
        lower_bound: left boundary of histogram values
        upper_bound: right boundary of histogram values
        num_buckets: number of buckets to use in [lower_bound, upper_bound)
        accumulate: boolean to output accumulate or per-batch histogram
    """

    def __init__(self, blobs, logging_frequency, num_buckets=30,
            lower_bound=0.0, upper_bound=1.0, accumulate=False):
        self._blobs = blobs
        self._logging_frequency = logging_frequency
        self._accumulate = accumulate
        if self._accumulate:
            self._field_name_suffix = '_acc_normalized_hist'
        else:
            self._field_name_suffix = '_curr_normalized_hist'

        self._num_buckets = int(num_buckets)
        assert self._num_buckets > 0, (
            "num_buckets need to be greater than 0, got {}".format(num_buckets))
        self._lower_bound = float(lower_bound)
        self._upper_bound = float(upper_bound)

    def modify_net(self, net, init_net=None, grad_map=None, blob_to_device=None,
                   modify_output_record=False):
        for blob_name in self._blobs:
            blob = core.BlobReference(blob_name)
            assert net.BlobIsDefined(blob), 'blob {} is not defined in net {} whose proto is {}'.format(blob, net.Name(), net.Proto())

            blob_float = net.Cast(blob, net.NextScopedBlob(prefix=blob +
                '_float'), to=core.DataType.FLOAT)
            curr_hist, acc_hist = net.AccumulateHistogram(
                [blob_float],
                [net.NextScopedBlob(prefix=blob + '_curr_hist'),
                 net.NextScopedBlob(prefix=blob + '_acc_hist')],
                num_buckets=self._num_buckets,
                lower_bound=self._lower_bound,
                upper_bound=self._upper_bound)

            if self._accumulate:
                hist = net.Cast(
                    acc_hist,
                    net.NextScopedBlob(prefix=blob + '_cast_hist'),
                    to=core.DataType.FLOAT)
            else:
                hist = net.Cast(
                    curr_hist,
                    net.NextScopedBlob(prefix=blob + '_cast_hist'),
                    to=core.DataType.FLOAT)

            normalized_hist = net.NormalizeL1(
                hist,
                net.NextScopedBlob(prefix=blob + self._field_name_suffix)
            )

            if self._logging_frequency >= 1:
                net.Print(normalized_hist, [], every_n=self._logging_frequency)

            if modify_output_record:
                output_field_name = str(blob) + self._field_name_suffix
                output_scalar = schema.Scalar((np.float32, (self._num_buckets + 2,)),
                    normalized_hist)

                if net.output_record() is None:
                    net.set_output_record(
                        schema.Struct((output_field_name, output_scalar))
                    )
                else:
                    net.AppendOutputRecordField(
                        output_field_name,
                        output_scalar)

    def field_name_suffix(self):
        return self._field_name_suffix
