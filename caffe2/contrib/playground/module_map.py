from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# Input
import caffe2.contrib.playground.resnet50demo.\
    gfs_IN1k as gfs_IN1k  # noqa

# model
import caffe2.contrib.playground.resnet50demo.\
    IN1k_resnet50 as IN1k_resnet50 # noqa

# FORWARD_PASS
import caffe2.contrib.playground.resnet50demo.\
    caffe2_resnet50_default_forward as caffe2_resnet50_default_forward # noqa

import caffe2.contrib.playground.resnet50demo.\
    explicit_resnet_forward as explicit_resnet_forward # noqa

# PARAMETER_UPDATE
import caffe2.contrib.playground.resnet50demo.\
    caffe2_resnet50_default_param_update as caffe2_resnet50_default_param_update # noqa

import caffe2.contrib.playground.resnet50demo.\
    explicit_resnet_param_update as explicit_resnet_param_update # noqa

# RENDEZVOUS
import caffe2.contrib.playground.resnet50demo.\
    rendezvous_filestore as rendezvous_filestore # noqa

# OUTPUT
import caffe2.contrib.playground.\
    output_generator as output_generator # noqa

# METERS
# for meters, use the class name as your module name in this map
import caffe2.contrib.playground.\
    compute_loss as ComputeLoss # noqa

import caffe2.contrib.playground.\
    compute_topk_accuracy as ComputeTopKAccuracy # noqa
