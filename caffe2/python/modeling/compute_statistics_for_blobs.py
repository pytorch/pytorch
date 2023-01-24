




from caffe2.python import core, schema
from caffe2.python.modeling.net_modifier import NetModifier

import numpy as np


class ComputeStatisticsForBlobs(NetModifier):
    """
    This class modifies the net passed in by adding ops to compute statistics
    for certain blobs. For each blob in the list, its min, max, mean and standard
    deviation will be computed.

    Args:
        blobs: list of blobs to compute norm for
        logging_frequency: frequency for printing norms to logs
    """

    def __init__(self, blobs, logging_frequency):
        self._blobs = blobs
        self._logging_frequency = logging_frequency
        self._field_name_suffix = '_summary'

    def modify_net(self, net, init_net=None, grad_map=None, blob_to_device=None,
                   modify_output_record=False):

        for blob_name in self._blobs:
            blob = core.BlobReference(blob_name)
            assert net.BlobIsDefined(blob), 'blob {} is not defined in net {} whose proto is {}'.format(blob, net.Name(), net.Proto())

            cast_blob = net.Cast(blob, to=core.DataType.FLOAT)
            stats_name = net.NextScopedBlob(prefix=blob + self._field_name_suffix)
            stats = net.Summarize(cast_blob, stats_name, to_file=0)
            net.Print(stats, [], every_n=self._logging_frequency)

            if modify_output_record:
                output_field_name = str(blob) + self._field_name_suffix
                output_scalar = schema.Scalar((np.float, (1,)), stats)

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
