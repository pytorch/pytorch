# ---[ CPU files.
file(GLOB tmp *.cc)
set(Caffe2_CPU_SRCS ${Caffe2_CPU_SRCS} ${tmp})

# ---[ Send the lists to the parent scope.
set(Caffe2_CPU_SRCS ${Caffe2_CPU_SRCS} PARENT_SCOPE)
