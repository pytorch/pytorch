## @package hip_test_util
# Module caffe2.python.hip_test_util
"""
The HIP test utils is a small addition on top of the hypothesis test utils
under caffe2/python, which allows one to more easily test HIP/ROCm related
operators.
"""






from caffe2.proto import caffe2_pb2

def run_in_hip(gc, dc):
    return (gc.device_type == caffe2_pb2.HIP) or (
        caffe2_pb2.HIP in {d.device_type for d in dc})
