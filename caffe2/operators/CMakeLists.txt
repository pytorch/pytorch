# ---[ GPU files
# ------[ cuDNN
if (USE_CUDNN)
  file(GLOB tmp *_cudnn.cc *_cudnn.cu)
  set(Caffe2_GPU_SRCS ${Caffe2_GPU_SRCS} ${tmp})
endif()
# ------[ general GPU
file(GLOB tmp *_gpu.cc)
set(Caffe2_GPU_SRCS ${Caffe2_GPU_SRCS} ${tmp})
# ------[ TensorRT
if (USE_TENSORRT)
file(GLOB tmp *_trt.cc)
set(Caffe2_GPU_SRCS ${Caffe2_GPU_SRCS} ${tmp})
endif()
# ------[ CUDA sources
file(GLOB tmp *.cu)
# Manually remove the cudnn files since we might be using USE_CUDNN=OFF
# TODO: when we move to explicit file list, this would not be needed.
file(GLOB tmp_cudnn *_cudnn.cu)
exclude(tmp "${tmp}" ${tmp_cudnn})
set(Caffe2_GPU_SRCS ${Caffe2_GPU_SRCS} ${tmp})
# exclude test files
file(GLOB tmp *_test.cc)
exclude(Caffe2_GPU_SRCS "${Caffe2_GPU_SRCS}" ${tmp})

# ------[ general HIP
file(GLOB tmp hip/*.cc)
set(Caffe2_HIP_SRCS ${Caffe2_HIP_SRCS} ${tmp})
# ------[ HIP sources
file(GLOB tmp hip/*.hip)
set(Caffe2_HIP_SRCS ${Caffe2_HIP_SRCS} ${tmp})
# exclude test files
file(GLOB tmp hip/*_test.cc)
exclude(Caffe2_HIP_SRCS "${Caffe2_HIP_SRCS}" ${tmp})

# ---[ CPU files.
file(GLOB tmp *.cc)
# Manually remove the cudnn files since we might be using USE_CUDNN=OFF
# TODO: when we move to explicit file list, this would not be needed.
file(GLOB tmp_cudnn *_cudnn.cc)
exclude(tmp "${tmp}" ${tmp_cudnn})
set(Caffe2_CPU_SRCS ${Caffe2_CPU_SRCS} ${tmp})

# Add all files in experimental
if (NOT INTERN_BUILD_MOBILE)
    set(Caffe2_CPU_SRCS ${Caffe2_CPU_SRCS} ${CMAKE_CURRENT_LIST_DIR}/experimental/c10/cpu/flatten_cpu.cc)
    set(Caffe2_CPU_SRCS ${Caffe2_CPU_SRCS} ${CMAKE_CURRENT_LIST_DIR}/experimental/c10/cpu/averaged_loss_cpu.cc)
    set(Caffe2_CPU_SRCS ${Caffe2_CPU_SRCS} ${CMAKE_CURRENT_LIST_DIR}/experimental/c10/cpu/mul_cpu.cc)
    set(Caffe2_CPU_SRCS ${Caffe2_CPU_SRCS} ${CMAKE_CURRENT_LIST_DIR}/experimental/c10/cpu/relu_cpu.cc)
    set(Caffe2_CPU_SRCS ${Caffe2_CPU_SRCS} ${CMAKE_CURRENT_LIST_DIR}/experimental/c10/cpu/expand_dims_cpu.cc)
    set(Caffe2_CPU_SRCS ${Caffe2_CPU_SRCS} ${CMAKE_CURRENT_LIST_DIR}/experimental/c10/cpu/filler_cpu.cc)
    set(Caffe2_CPU_SRCS ${Caffe2_CPU_SRCS} ${CMAKE_CURRENT_LIST_DIR}/experimental/c10/cpu/sparse_lengths_sum_cpu.cc)
    set(Caffe2_CPU_SRCS ${Caffe2_CPU_SRCS} ${CMAKE_CURRENT_LIST_DIR}/experimental/c10/cpu/sigmoid_cpu.cc)
    set(Caffe2_CPU_SRCS ${Caffe2_CPU_SRCS} ${CMAKE_CURRENT_LIST_DIR}/experimental/c10/cpu/cast_cpu.cc)
    set(Caffe2_CPU_SRCS ${Caffe2_CPU_SRCS} ${CMAKE_CURRENT_LIST_DIR}/experimental/c10/cpu/stop_gradient_cpu.cc)
    set(Caffe2_CPU_SRCS ${Caffe2_CPU_SRCS} ${CMAKE_CURRENT_LIST_DIR}/experimental/c10/cpu/batch_gather_cpu.cc)
    set(Caffe2_CPU_SRCS ${Caffe2_CPU_SRCS} ${CMAKE_CURRENT_LIST_DIR}/experimental/c10/cpu/concat_cpu.cc)
    set(Caffe2_CPU_SRCS ${Caffe2_CPU_SRCS} ${CMAKE_CURRENT_LIST_DIR}/experimental/c10/cpu/batch_matmul_cpu.cc)
    set(Caffe2_CPU_SRCS ${Caffe2_CPU_SRCS} ${CMAKE_CURRENT_LIST_DIR}/experimental/c10/cpu/sigmoid_cross_entropy_with_logits_cpu.cc)
    set(Caffe2_CPU_SRCS ${Caffe2_CPU_SRCS} ${CMAKE_CURRENT_LIST_DIR}/experimental/c10/cpu/fc_cpu.cc)
    set(Caffe2_CPU_SRCS ${Caffe2_CPU_SRCS} ${CMAKE_CURRENT_LIST_DIR}/experimental/c10/cpu/enforce_finite_cpu.cc)
    set(Caffe2_CPU_SRCS ${Caffe2_CPU_SRCS} ${CMAKE_CURRENT_LIST_DIR}/experimental/c10/cpu/add_cpu.cc)
endif()

set(Caffe2_CPU_SRCS ${Caffe2_CPU_SRCS} ${tmp})
# exclude test files and gpu files
file(GLOB tmp *_test.cc)
exclude(Caffe2_CPU_SRCS "${Caffe2_CPU_SRCS}" ${tmp})
exclude(Caffe2_CPU_SRCS "${Caffe2_CPU_SRCS}" ${Caffe2_GPU_SRCS} ${Caffe2_HIP_SRCS})

# ---[ GPU test files
# ------[ cuDNN
if (USE_CUDNN)
  file(GLOB tmp *_cudnn_test.cc)
  set(Caffe2_GPU_TEST_SRCS ${Caffe2_GPU_TEST_SRCS} ${tmp})
endif()
# ------[ general GPU
file(GLOB tmp *_gpu_test.cc)
set(Caffe2_GPU_TEST_SRCS ${Caffe2_GPU_TEST_SRCS} ${tmp})

# ---[ HIP test files
file(GLOB tmp hip/*_test.cc)
set(Caffe2_HIP_TEST_SRCS ${Caffe2_HIP_TEST_SRCS} ${tmp})

# ---[ CPU test files
file(GLOB tmp *_test.cc)
# Manually remove the cudnn files since we might be using USE_CUDNN=OFF
# TODO: when we move to explicit file list, this would not be needed.
file(GLOB tmp_cudnn *_cudnn_test.cc)
exclude(tmp "${tmp}" ${tmp_cudnn})
set(Caffe2_CPU_TEST_SRCS ${Caffe2_CPU_TEST_SRCS} ${tmp})
exclude(Caffe2_CPU_TEST_SRCS "${Caffe2_CPU_TEST_SRCS}" ${Caffe2_GPU_TEST_SRCS} ${Caffe2_HIP_TEST_SRCS})

# ---[ Send the lists to the parent scope.
set(Caffe2_CPU_SRCS ${Caffe2_CPU_SRCS} PARENT_SCOPE)
set(Caffe2_GPU_SRCS ${Caffe2_GPU_SRCS} PARENT_SCOPE)
set(Caffe2_HIP_SRCS ${Caffe2_HIP_SRCS} PARENT_SCOPE)
set(Caffe2_CPU_TEST_SRCS ${Caffe2_CPU_TEST_SRCS} PARENT_SCOPE)
set(Caffe2_GPU_TEST_SRCS ${Caffe2_GPU_TEST_SRCS} PARENT_SCOPE)
set(Caffe2_HIP_TEST_SRCS ${Caffe2_HIP_TEST_SRCS} PARENT_SCOPE)
