




import sys
import numpy as np

def print_test_debug_info(testname, items_dict):
    filename = "debug_operator_onnxifi_" + testname + ".txt"
    np.set_printoptions(threshold=sys.maxsize)
    with open(filename, 'w') as f:
        for key, value in items_dict.items():
            print(key, value)
            f.write("{}\n".format(key))
            f.write("{}\n".format(value))

def print_net(net):
    for i in net.external_input:
        print("Input: {}".format(i))
    for i in net.external_output:
        print("Output: {}".format(i))
    for op in net.op:
        print("Op {}".format(op.type))
        for x in op.input:
            print("  input: {}".format(x))
        for y in op.output:
            print("  output: {}".format(y))

def _sigmoid(x):
    return 1. / (1. + np.exp(np.float64(-x)))

def _tanh(x):
    return np.tanh(np.float64(x))

def _swish(x):
    return np.float64(x) * _sigmoid(x)

def _gelu_by_sigmoid(x):
    return np.float64(x) / (1. + np.exp(np.float64(x) * 1.702))


def _acc_func(opname, x):
    if opname == "Swish":
        return _swish(x)
    elif opname == "Sigmoid":
        return _sigmoid(x)
    elif opname == "Tanh":
        return _tanh(x)
    elif opname == "Gelu":
        return _gelu_by_sigmoid(x)
    else:
        return x

def _get_ulp16(x):
    abs_x = np.abs(x)
    mask = (abs_x > 2.**(-14))
    abs_x = mask * abs_x + (1 - mask) * 2.**(-14)
    k = np.floor(np.log2(abs_x))
    return 2.**(k - 10)

def compute_ulp_error(opname, xvec, y_nnpi):
    y_acc = _acc_func(opname, np.float64(xvec))
    scale = 1. / _get_ulp16(y_acc)
    return (y_nnpi - y_acc) * scale
