#!/bin/bash

# Split up the doxygen run into multiple runs so that we can set different
# INPUT and XML_OUTPUT values for each run. This way we can have more control

# ATen
doxygen - <<EOF
$(cat Doxyfile)
INPUT                 =  ../../../aten/src/ATen/ATen.h \
                         ../../../aten/src/ATen/Backend.h \
                         ../../../aten/src/ATen/core/ivalue.h \
                         ../../../aten/src/ATen/core/ScalarType.h \
                         ../../../aten/src/ATen/cuda/CUDAContext.h \
                         ../../../aten/src/ATen/cudnn/Descriptors.h \
                         ../../../aten/src/ATen/cudnn/Handles.h \
                         ../../../aten/src/ATen/cudnn/Types.h \
                         ../../../aten/src/ATen/cudnn/Utils.h \
                         ../../../aten/src/ATen/DeviceGuard.h \
                         ../../../aten/src/ATen/Layout.h \
                         ../../../aten/src/ATen/mkl/Descriptors.h \
                         ../../../aten/src/ATen/Scalar.h \
                         ../../../aten/src/ATen/TensorOptions.h \
                         ../../../aten/src/ATen/core/Tensor.h \
                         ../../../aten/src/ATen/native/TensorShape.h \
                         ../../../build/aten/src/ATen/Functions.h \
                         ../../../build/aten/src/ATen/core/TensorBody.h
XML_OUTPUT = xml/aten
EOF

# c10
doxygen - <<EOF
$(cat Doxyfile)
INPUT                 =  ../../../c10/core/Device.h \
                         ../../../c10/core/DeviceType.h \
                         ../../../c10/util/Half.h \
                         ../../../c10/util/ArrayRef.h \
                         ../../../c10/util/OptionalArrayRef.h \
                         ../../../c10/util/Exception.h \
                         ../../../c10/util/Optional.h \
                         ../../../c10/cuda/CUDAGuard.h \
                         ../../../c10/cuda/CUDAStream.h \
                         ../../../c10/xpu/XPUStream.h
XML_OUTPUT = xml/c10
EOF


# csrc
doxygen - <<EOF
$(cat Doxyfile)
INPUT                 =  ../../../torch/csrc/api/include \
                         ../../../torch/csrc/api/src \
                         ../../../torch/csrc/autograd/autograd.h \
                         ../../../torch/csrc/autograd/custom_function.h \
                         ../../../torch/csrc/autograd/function.h \
                         ../../../torch/csrc/autograd/variable.h \
                         ../../../torch/csrc/autograd/generated/variable_factories.h \
                         ../../../torch/csrc/jit/runtime/custom_operator.h \
                         ../../../torch/csrc/jit/serialization/import.h \
                         ../../../torch/csrc/jit/api/module.h \
                         ../../../torch/csrc/stable/library.h
XML_OUTPUT = xml/csrc
EOF



# other
doxygen - <<EOF
$(cat Doxyfile)
INPUT                 =  ../../../torch/library.h \
                         ../../../torch/custom_class.h
XML_OUTPUT = xml/csrc
EOF
