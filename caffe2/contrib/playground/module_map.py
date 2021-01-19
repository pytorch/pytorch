




# Input
import caffe2.contrib.playground.resnetdemo.\
    gfs_IN1k as gfs_IN1k  # noqa

# model
import caffe2.contrib.playground.resnetdemo.\
    IN1k_resnet as IN1k_resnet # noqa

import caffe2.contrib.playground.resnetdemo.\
    IN1k_resnet_no_test_model as IN1k_resnet_no_test_model # noqa

# Additional override
import caffe2.contrib.playground.resnetdemo.\
    override_no_test_model_no_checkpoint as override_no_test_model_no_checkpoint # noqa

# FORWARD_PASS
import caffe2.contrib.playground.resnetdemo.\
    caffe2_resnet50_default_forward as caffe2_resnet50_default_forward # noqa

import caffe2.contrib.playground.resnetdemo.\
    explicit_resnet_forward as explicit_resnet_forward # noqa

# PARAMETER_UPDATE
import caffe2.contrib.playground.resnetdemo.\
    caffe2_resnet50_default_param_update as caffe2_resnet50_default_param_update # noqa

import caffe2.contrib.playground.resnetdemo.\
    explicit_resnet_param_update as explicit_resnet_param_update # noqa

# RENDEZVOUS
import caffe2.contrib.playground.resnetdemo.\
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
