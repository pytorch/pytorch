// Copyright (c) Facebook Inc. and Microsoft Corporation.
// Licensed under the MIT license.

#include "./schema.h"

namespace ONNX_NAMESPACE {

ONNX_PYTORCH_OPERATOR_SET_SCHEMA(
    SparseLengthsSumFused8BitRowwise,
    1,
    OpSchema()
        .SetDoc("Mirror Caffe2 SparseLengthsSumFused8BitRowwise operator")
        .Input(0, "DATA", "data tensor", "T1")
        .Input(1, "INDICES", "indices tensor", "T2")
        .Input(2, "LENGTHS", "lengths tensor", "T2")
        .Output(0, "output", "Output tensor", "T2")
        .TypeConstraint(
            "T1",
            {"tensor(uint8)"},
            "Constrain input data to uint8 tensors.")
        .TypeConstraint(
            "T2",
            {"tensor(int8)",
             "tensor(int16)",
             "tensor(int32)",
             "tensor(int64)",
             "tensor(uint8)",
             "tensor(uint16)",
             "tensor(uint32)",
             "tensor(uint64)"},
            "Constrain index and length to integral tensors."));

ONNX_PYTORCH_OPERATOR_SET_SCHEMA(
    SparseLengthsSum,
    1,
    OpSchema()
        .SetDoc("Mirror Caffe2 SparseLengthsSum operator")
        .Input(0, "DATA", "data tensor", "T1")
        .Input(1, "INDICES", "indices tensor", "T2")
        .Input(2, "LENGTHS", "lengths tensor", "T2")
        .Output(0, "output", "Output tensor", "T1")
        .TypeConstraint(
            "T1",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors.")
        .TypeConstraint(
            "T2",
            {"tensor(int8)",
             "tensor(int16)",
             "tensor(int32)",
             "tensor(int64)",
             "tensor(uint8)",
             "tensor(uint16)",
             "tensor(uint32)",
             "tensor(uint64)"},
            "Constrain index and length to integral tensors."));

ONNX_PYTORCH_OPERATOR_SET_SCHEMA(
    SparseLengthsWeightedSum,
    1,
    OpSchema()
        .SetDoc("Mirror Caffe2 SparseLengthsWeightedSum operator")
        .Input(0, "DATA", "data tensor", "T1")
        .Input(1, "WEIGHTS", "data tensor", "T1")
        .Input(2, "INDICES", "indices tensor", "T2")
        .Input(3, "LENGTHS", "lengths tensor", "T2")
        .Output(0, "output", "Output tensor", "T1")
        .TypeConstraint(
            "T1",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors.")
        .TypeConstraint(
            "T2",
            {"tensor(int8)",
             "tensor(int16)",
             "tensor(int32)",
             "tensor(int64)",
             "tensor(uint8)",
             "tensor(uint16)",
             "tensor(uint32)",
             "tensor(uint64)"},
            "Constrain index and length to integral tensors."));

ONNX_PYTORCH_OPERATOR_SET_SCHEMA(
    BatchGather,
    1,
    OpSchema()
        .SetDoc("Mirror Caffe2 BatchGather operator")
        .Input(0, "DATA", "data tensor", "T1")
        .Input(1, "INDICES", "indices tensor", "T2")
        .Output(0, "output", "Output tensor", "T1")
        .TypeConstraint(
            "T1",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors.")
        .TypeConstraint(
            "T2",
            {"tensor(int8)",
             "tensor(int16)",
             "tensor(int32)",
             "tensor(int64)",
             "tensor(uint8)",
             "tensor(uint16)",
             "tensor(uint32)",
             "tensor(uint64)"},
            "Constrain index and length to integral tensors."));

ONNX_PYTORCH_OPERATOR_SET_SCHEMA(
    DotProduct,
    1,
    OpSchema()
        .SetDoc("Mirror Caffe2 DotProduct operator")
        .Input(0, "X", "Input 1 tensor", "T")
        .Input(1, "Y", "Input 2 tensor", "T")
        .Output(0, "Z", "Output tensor", "T")
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors."));

ONNX_PYTORCH_OPERATOR_SET_SCHEMA(
    FCTransposed,
    1,
    OpSchema()
        .SetDoc("Mirror Caffe2 FCTransposed operator")
        .Input(0, "X", "Input tensor", "T")
        .Input(1, "W", "Weight tensor", "T")
        .Input(2, "B", "Bias tensor", "T")
        .Output(0, "Z", "Output tensor", "T")
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors."));

ONNX_PYTORCH_OPERATOR_SET_SCHEMA(
    BatchMatMul,
    1,
    OpSchema()
        .SetDoc("Mirror Caffe2 BatchMatMul operator")
        .Input(0, "X", "tensor of shape (dim0, dim1 ... M, K)", "T")
        .Input(1, "Y", "tensor of shape (dim0, dim2 ... K, N)", "T")
        .Output(0, "Z", "tensor of shape (dim0, dim1 ... M, N)", "T")
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors."));

ONNX_PYTORCH_OPERATOR_SET_SCHEMA(
    ExpandDims,
    1,
    OpSchema()
        .SetDoc("Mirror Caffe2 ExpandDims operator")
        .Input(0, "X", "Input tensor", "T")
        .Output(0, "Y", "Output tensor", "T")
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors."));

} // namespace ONNX_NAMESPACE
