set(DEPLOY_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
add_subdirectory(interpreter)

# we do not want to have torch_deployinterpreter linked against libstdc++ or libc because
# when loading it with RTLD_DEEPBIND it will resolve std::cout/stdout to the copy in libc++/libc instead of the
# ones in the main process (see https://gcc.gnu.org/bugzilla/show_bug.cgi?id=42679).
# However, we can't just instruct the linker to not link against these libraries because these
# libraries use function versioning. Without linking them, the shared library would not know the right
# symbol versions and instead try to link against the old ones. Our solution is to link the library
# normally then remove the DT_NEEDED entries in the ELF file that instruct the loaded to load the sublibraries.
# This gives us the right version numbers but no direct dependency on libstdc++/libc. When loaded these
# symbols will fallback to resolution through the main execution and get the correct values
add_executable(remove_dt_needed remove_dt_needed.cpp)
target_link_libraries(remove_dt_needed PRIVATE fmt::fmt-header-only)

add_custom_command(
  OUTPUT libtorch_deployinterpreter.o
  # remove the DT_NEEDED entries
  COMMAND $<TARGET_FILE:remove_dt_needed> $<TARGET_FILE:torch_deployinterpreter> libtorch_deployinterpreter_all.so
  # package the result into an object we can link into the libdeploy binary.
  COMMAND ld -r -b binary -o libtorch_deployinterpreter.o libtorch_deployinterpreter_all.so
  COMMAND objcopy --rename-section .data=.torch_deploy_payload.interpreter_all,readonly,contents -N _binary_libtorch_deployinterpreter_all_so_start -N _binary_libtorch_deployinterpreter_all_so_end libtorch_deployinterpreter.o
  COMMAND rm libtorch_deployinterpreter_all.so
  DEPENDS torch_deployinterpreter remove_dt_needed
  VERBATIM
)

add_library(torch_deploy_internal STATIC libtorch_deployinterpreter.o ${DEPLOY_DIR}/deploy.cpp ${DEPLOY_DIR}/loader.cpp ${DEPLOY_DIR}/path_environment.cpp ${DEPLOY_DIR}/elf_file.cpp)
target_link_libraries(torch_deploy_internal PRIVATE crypt pthread dl util m z ffi lzma readline nsl ncursesw panelw) # for python builtins
target_link_libraries(torch_deploy_internal PUBLIC  shm torch fmt::fmt-header-only protobuf::libprotobuf-lite)
caffe2_interface_library(torch_deploy_internal torch_deploy)

set(INTERPRETER_TEST_SOURCES
  ${DEPLOY_DIR}/test_deploy.cpp
)
set(INTERPRETER_TEST_SOURCES_GPU
  ${DEPLOY_DIR}/test_deploy_gpu.cpp
)

add_executable(test_deploy ${INTERPRETER_TEST_SOURCES})
target_compile_definitions(test_deploy PUBLIC TEST_CUSTOM_LIBRARY)
target_include_directories(test_deploy PRIVATE ${PYTORCH_ROOT}/torch)
target_link_libraries(test_deploy PUBLIC "-Wl,--no-as-needed" gtest dl torch_deploy)

add_executable(test_deploy_gpu ${INTERPRETER_TEST_SOURCES_GPU})
target_compile_definitions(test_deploy_gpu PUBLIC TEST_CUSTOM_LIBRARY)
target_include_directories(test_deploy_gpu PRIVATE ${PYTORCH_ROOT}/torch)
target_link_libraries(test_deploy_gpu PUBLIC "-Wl,--no-as-needed" gtest dl torch_deploy)

add_library(test_deploy_lib SHARED test_deploy_lib.cpp)
add_dependencies(test_deploy_lib cpython)
target_include_directories(test_deploy_lib BEFORE PRIVATE ${PYTHON_INC_DIR})
target_link_libraries(test_deploy_lib PRIVATE pybind::pybind11)

add_executable(deploy_benchmark ${DEPLOY_DIR}/example/benchmark.cpp)
target_include_directories(deploy_benchmark PRIVATE ${PYTORCH_ROOT}/torch)
target_link_libraries(deploy_benchmark PUBLIC "-Wl,--no-as-needed" torch_deploy)

add_executable(interactive_embedded_interpreter ${DEPLOY_DIR}/interactive_embedded_interpreter.cpp)
target_include_directories(interactive_embedded_interpreter PRIVATE ${PYTORCH_ROOT}/torch)
target_link_libraries(interactive_embedded_interpreter PUBLIC "-Wl,--no-as-needed" torch_deploy)

if(INSTALL_TEST)
  install(TARGETS test_deploy DESTINATION bin)
  install(TARGETS test_deploy_gpu DESTINATION bin)
endif()

install(TARGETS torch_deploy DESTINATION lib)
