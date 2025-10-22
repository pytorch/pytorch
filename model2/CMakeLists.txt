cmake_minimum_required(VERSION 3.27 FATAL_ERROR)
project(aoti_example LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


add_executable(aoti_example main.cpp)
set_property(TARGET aoti_example PROPERTY CXX_STANDARD 17)

find_package(TorchStandalone REQUIRED)
# Set up include directories to find headers at the correct paths
target_include_directories(aoti_example PRIVATE ${TorchStandalone_INCLUDE_DIRS})
target_include_directories(aoti_example PRIVATE ${TorchStandalone_INCLUDE_DIRS}/standalone)

enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
find_package(CUDAToolkit REQUIRED)


target_compile_definitions(aoti_example PRIVATE NOMINMAX USE_CUDA)

# Add compile flags
target_compile_options(aoti_example PRIVATE /O2 /DLL /MD /std:c++20 /wd4819 /wd4251 /wd4244 /wd4267 /wd4275 /wd4018 /wd4190 /wd4624 /wd4067 /wd4068 /EHsc /Zc:__cplusplus /permissive- /openmp /openmp:experimental )


target_link_libraries(aoti_example PRIVATE
    ${TorchStandalone_LIBRARIES}  # if you have this variable from find_package(TorchStandalone)
    CUDA::cudart                  # CUDA runtime library
    cuda                       # CUDA driver library (usually nvcuda.lib on Windows)
)

# cmake -DTorchStandalone_DIR="C:/Users/shangdiy/source/repos/torchnative/standalone/build/torchstandalone_install/lib/cmake/TorchStandalone"  ..  
# cmake --build . --config Release