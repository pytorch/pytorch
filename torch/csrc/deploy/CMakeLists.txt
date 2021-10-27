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
  COMMAND rm libtorch_deployinterpreter_all.so
  DEPENDS torch_deployinterpreter remove_dt_needed
  VERBATIM
)

add_library(torch_deploy libtorch_deployinterpreter.o ${DEPLOY_DIR}/deploy.cpp ${DEPLOY_DIR}/loader.cpp)
target_link_libraries(torch_deploy PRIVATE crypt pthread dl util m z ffi lzma readline nsl ncursesw panelw) # for python builtins
target_link_libraries(torch_deploy PUBLIC "-Wl,--no-as-needed" shm torch fmt::fmt-header-only protobuf::libprotobuf-lite)


set(INTERPRETER_TEST_SOURCES
  ${DEPLOY_DIR}/test_deploy.cpp
)
add_executable(test_deploy ${INTERPRETER_TEST_SOURCES})
target_compile_definitions(test_deploy PUBLIC TEST_CUSTOM_LIBRARY)
target_include_directories(test_deploy PRIVATE ${PYTORCH_ROOT}/torch)
target_link_libraries(test_deploy PUBLIC gtest dl torch_deploy)

add_library(test_deploy_lib SHARED test_deploy_lib.cpp)
add_dependencies(test_deploy_lib cpython)
target_include_directories(test_deploy_lib BEFORE PRIVATE ${PYTHON_INC_DIR})

add_executable(deploy_benchmark ${DEPLOY_DIR}/example/benchmark.cpp)
target_include_directories(deploy_benchmark PRIVATE ${PYTORCH_ROOT}/torch)
target_link_libraries(deploy_benchmark PUBLIC torch_deploy)

add_executable(interactive_embedded_interpreter ${DEPLOY_DIR}/interactive_embedded_interpreter.cpp)
target_include_directories(interactive_embedded_interpreter PRIVATE ${PYTORCH_ROOT}/torch)
target_link_libraries(interactive_embedded_interpreter PUBLIC torch_deploy)

if(INSTALL_TEST)
  install(TARGETS test_deploy DESTINATION bin)
  install(TARGETS torch_deploy DESTINATION lib)
endif()
