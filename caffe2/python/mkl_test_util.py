## @package mkl_test_util
# Module caffe2.python.mkl_test_util
"""
The MKL test utils is a small addition on top of the hypothesis test utils
under caffe2/python, which allows one to more easily test MKL related
operators.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import hypothesis.strategies as st

from caffe2.proto import caffe2_pb2
from caffe2.python import workspace
from caffe2.python import hypothesis_test_util as hu

cpu_do = hu.cpu_do
gpu_do = hu.gpu_do
mkl_do = caffe2_pb2.DeviceOption(device_type=caffe2_pb2.MKLDNN)
device_options = hu.device_options + (
    [mkl_do] if workspace.C.has_mkldnn else [])


def device_checker_device_options():
    return st.just(device_options)


def gradient_checker_device_option():
    return st.sampled_from(device_options)


gcs = dict(
    gc=gradient_checker_device_option(),
    dc=device_checker_device_options()
)

gcs_cpu_only = dict(gc=st.sampled_from([cpu_do]), dc=st.just([cpu_do]))
gcs_gpu_only = dict(gc=st.sampled_from([gpu_do]), dc=st.just([gpu_do]))
gcs_mkl_only = dict(gc=st.sampled_from([mkl_do]), dc=st.just([mkl_do]))

gcs_cpu_mkl = dict(gc=st.sampled_from([cpu_do, mkl_do]), dc=st.just([cpu_do, mkl_do]))
