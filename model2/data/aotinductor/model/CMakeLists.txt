
cmake_minimum_required(VERSION 3.27 FATAL_ERROR)
project(model LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

# Set a library target
add_library(model SHARED)


# TODO: change to TorchStandalone
find_package(TorchStandalone REQUIRED)
# Set up include directories to find headers at the correct paths
target_include_directories(model PRIVATE ${TorchStandalone_INCLUDE_DIRS})
target_include_directories(model PRIVATE ${TorchStandalone_INCLUDE_DIRS}/standalone)


# Add macro definitions
target_compile_definitions(model PRIVATE NOMINMAX TORCH_INDUCTOR_CPP_WRAPPER STANDALONE_TORCH_HEADER  C10_USING_CUSTOM_GENERATED_MACROS USE_CUDA) # CPU_CAPABILITY_AVX512  

# Add compile flags
target_compile_options(model PRIVATE /O2 /DLL /MD /std:c++20 /wd4819 /wd4251 /wd4244 /wd4267 /wd4275 /wd4018 /wd4190 /wd4624 /wd4067 /wd4068 /EHsc /Zc:__cplusplus /permissive- /openmp /openmp:experimental )

# Backend-specific flags
# target_compile_options(model PRIVATE  -mavx512f -mavx512dq -mavx512vl -mavx512bw -mfma -mavx512bf16  -c)  # TODO remove


enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
find_package(CUDAToolkit REQUIRED)

# Make output use .pyd instead of .dll
set_target_properties(model PROPERTIES 
    SUFFIX ".pyd" 
    LINK_FLAGS "/DEF:${CMAKE_CURRENT_SOURCE_DIR}/model_exports.def"
)

set(KERNEL_TARGETS "")
set(KERNEL_OBJECT_FILES "")
# Function to compile ptx to cubin
function(embed_gpu_kernel KERNEL_NAME PTX_FILE)
    set(CUBIN_BASENAME ${KERNEL_NAME}.cubin)
    set(CUBIN_FILE ${CMAKE_CURRENT_BINARY_DIR}/${CUBIN_BASENAME})
    # --- PTX to FATBIN Command & Target ---
    add_custom_command(
        OUTPUT ${CUBIN_FILE}
        COMMAND ${CUDAToolkit_NVCC_EXECUTABLE} --cubin ${PTX_FILE}
                -o ${CUBIN_FILE} ${NVCC_GENCODE_FLAGS}
                -gencode arch=compute_89,code=sm_89
        DEPENDS ${PTX_FILE}
    )

    add_custom_target(build_kernel_object_${KERNEL_NAME} DEPENDS ${CUBIN_FILE})
    set(KERNEL_TARGETS ${KERNEL_TARGETS} build_kernel_object_${KERNEL_NAME} PARENT_SCOPE)
endfunction()
target_sources(model PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/model.wrapper.cpp)
target_sources(model PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/model_consts.weights.cpp)

embed_gpu_kernel(model_triton_tem_fused_addmm_relu_t_0 ${CMAKE_CURRENT_SOURCE_DIR}/model_triton_tem_fused_addmm_relu_t_0.ptx)

embed_gpu_kernel(model_triton_tem_fused_addmm_relu_sigmoid_t_1 ${CMAKE_CURRENT_SOURCE_DIR}/model_triton_tem_fused_addmm_relu_sigmoid_t_1.ptx)
add_dependencies(model ${KERNEL_TARGETS})
target_link_libraries(model PRIVATE ${KERNEL_OBJECT_FILES})

# Add linker flags
target_link_options(model PRIVATE )

# Add libraries
# TODO: change to TorchStandalone
target_link_libraries(model PRIVATE  ${TorchStandalone_LIBRARIES} cuda CUDA::cudart)
