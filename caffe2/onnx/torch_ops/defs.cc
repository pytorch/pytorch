// Copyright (c) Facebook Inc. and Microsoft Corporation.
// Licensed under the MIT license.

#include "./schema.h"

namespace ONNX_NAMESPACE {

static const char* dummy_test_only_ver1_doc = R"DOC(
A dummy op for verifying the build setup works, don't use me.
)DOC";

ONNX_PYTORCH_OPERATOR_SET_SCHEMA(
    DUMMY_TEST_ONLY,
    1,
    OpSchema()
        .SetDoc(dummy_test_only_ver1_doc)
        .Input(0, "input", "Input tensor", "T")
        .Output(0, "output", "Output tensor", "T")
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors."));

} // namespace ONNX_NAMESPACE
