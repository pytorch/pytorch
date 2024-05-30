# ---[ Extra onnx files.
file(GLOB ONNX_SRCS *.cc)

# ---[ Send the lists to the parent scope.
set(ONNX_SRCS ${ONNX_SRCS} PARENT_SCOPE)
