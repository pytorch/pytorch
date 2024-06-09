# ---[ GPU files.
file(GLOB tmp *.cc)
set(Caffe2_GPU_SRCS ${Caffe2_GPU_SRCS} ${tmp})

# ---[ Send the lists to the parent scope.
set(Caffe2_GPU_SRCS ${Caffe2_GPU_SRCS} PARENT_SCOPE)
