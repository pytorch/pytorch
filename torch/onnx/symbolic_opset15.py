# EDITING THIS FILE? READ THIS FIRST!
# see Note [Edit Symbolic Files] in symbolic_helper.py

# This file exports ONNX ops for opset 15

# Note [ONNX operators that are added/updated in opset 15]
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# https://github.com/onnx/onnx/blob/master/docs/Changelog.md#version-15-of-the-default-onnx-operator-set
# New operators:
#   Bernoulli
#   CastLike
#   Optional
#   OptionalGetElement
#   OptionalHasElement
#
# Updated operators:
#    BatchNormalization https://github.com/onnx/onnx/pull/3545
#                       Backwards compatible
#                       TODO: test coverage for mixed types inputs.
#    Pow                https://github.com/onnx/onnx/pull/3412
#                       Backwards compatible
#                       TODO: bfloat16 support.
#    Shape              https://github.com/onnx/onnx/pull/3580
#                       Backwards compatible
#                       TODO: optional start/end attribute.
