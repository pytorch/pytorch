from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core, schema
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
        compute_averaged_norm: norm or averaged_norm (averaged_norm = norm/size)
    """

    def __init__(self, blobs, logging_frequency, p=2, compute_averaged_norm=False):
        self._blobs = blobs
        self._logging_frequency = logging_frequency
        self._p = p
        self._compute_averaged_norm = compute_averaged_norm
        self._field_name_suffix = '_l{}_norm'.format(p)
        if compute_averaged_norm:
            self._field_name_suffix = '_averaged' + self._field_name_suffix

    def modify_net(self, net, init_net=None, grad_map=None, blob_to_device=None):

        p = self._p
        compute_averaged_norm = self._compute_averaged_norm

        for blob_name in self._blobs:
            blob = core.BlobReference(blob_name)
            if not net.BlobIsDefined(blob):
                raise Exception('blob {0} is not defined in net {1}'.format(
                    blob, net.Name()))

            norm_name = net.NextScopedBlob(prefix=blob + self._field_name_suffix)
            norm = net.LpNorm(blob, norm_name, p=p, average=compute_averaged_norm)

            if self._logging_frequency >= 1:
                net.Print(norm, [], every_n=self._logging_frequency)

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
