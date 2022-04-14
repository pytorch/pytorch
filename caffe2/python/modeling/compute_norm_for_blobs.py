




from caffe2.python import core, schema, muji
from caffe2.python.modeling.net_modifier import NetModifier


import numpy as np


class ComputeNormForBlobs(NetModifier):
    """
    This class modifies the net passed in by adding ops to compute norms for
    certain blobs.

    Args:
        blobs: list of blobs to compute norm for
        logging_frequency: frequency for printing norms to logs
        p: type of norm. Currently it supports p=1 or p=2
        compute_averaged_norm: norm or averaged_norm (averaged_norm = norm/size
        row_index: to plot the entire blob or simply one row at the row_index)
    """

    def __init__(self, blobs, logging_frequency, p=2, compute_averaged_norm=False, row_index=None):
        self._blobs = blobs
        self._logging_frequency = logging_frequency
        self._p = p
        self._compute_averaged_norm = compute_averaged_norm
        self._field_name_suffix = '_l{}_norm'.format(p)
        if compute_averaged_norm:
            self._field_name_suffix = '_averaged' + self._field_name_suffix

        if row_index and row_index < 0:
            raise Exception('{0} is not a valid row index, row_index should be >= 0'.format(
                row_index))
        self.row_index = row_index

    def modify_net(self, net, init_net=None, grad_map=None, blob_to_device=None,
                   modify_output_record=False):

        p = self._p
        compute_averaged_norm = self._compute_averaged_norm
        row_index = self.row_index

        CPU = muji.OnCPU()
        # if given, blob_to_device is a map from blob to device_option
        blob_to_device = blob_to_device or {}
        for blob_name in self._blobs:
            blob = core.BlobReference(blob_name)
            assert net.BlobIsDefined(blob), 'blob {} is not defined in net {} whose proto is {}'.format(blob, net.Name(), net.Proto())
            if blob in blob_to_device:
                device = blob_to_device[blob]
            else:
                device = CPU

            with core.DeviceScope(device):
                if row_index and row_index >= 0:
                    blob = net.Slice(
                        [blob],
                        net.NextScopedBlob(prefix=blob + '_row_{0}'.format(row_index)),
                        starts=[row_index, 0],
                        ends=[row_index + 1, -1]
                    )

                cast_blob = net.Cast(
                    blob,
                    net.NextScopedBlob(prefix=blob + '_float'),
                    to=core.DataType.FLOAT
                )

                norm_name = net.NextScopedBlob(prefix=blob + self._field_name_suffix)
                norm = net.LpNorm(
                    cast_blob, norm_name, p=p, average=compute_averaged_norm
                )
                norm_stop_gradient = net.StopGradient(norm, net.NextScopedBlob(norm_name + "_stop_gradient"))

                if self._logging_frequency >= 1:
                    net.Print(norm, [], every_n=self._logging_frequency)

                if modify_output_record:
                    output_field_name = str(blob) + self._field_name_suffix
                    output_scalar = schema.Scalar((np.float, (1,)), norm)

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
