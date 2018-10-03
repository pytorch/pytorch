from caffe2.python import core, workspace, caffe2_pb2
import numpy as np


def run_add5_and_add5gradient_op(device):
    # clear the workspace before running the operator
    workspace.ResetWorkspace()
    add5 = core.CreateOperator("Add5",
                               ["X"],
                               ["Y"],
                               device_option=device)
    print("==> Running Add5 op:")
    workspace.FeedBlob("X", (np.random.rand(5, 5)), device_option=device)
    print("Input of Add5: ", workspace.FetchBlob("X"))
    workspace.RunOperatorOnce(add5)
    print("Output of Add5: ", workspace.FetchBlob("Y"))

    print("\n\n==> Running Add5Gradient op:")
    print("Input of Add5Gradient: ", workspace.FetchBlob("Y"))
    add5gradient = core.CreateOperator("Add5Gradient",
                                       ["Y"],
                                       ["Z"],
                                       device_option=device)
    workspace.RunOperatorOnce(add5gradient)
    print("Output of Add5Gradient: ", workspace.FetchBlob("Z"))


def main():
    # try device_type=caffe2_pb2.CUDA if CUDA is available in our build
    device = caffe2_pb2.DeviceOption(device_type=caffe2_pb2.CPU)
    run_add5_and_add5gradient_op(device)


if __name__ == "__main__":
    main()
