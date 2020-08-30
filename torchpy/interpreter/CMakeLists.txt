SET(INTERPRETER_DIR "${TORCHPY_DIR}/interpreter" )
SET(INTERPRETER_DIR "${TORCHPY_DIR}/interpreter" PARENT_SCOPE)

# Define verbose lists of objects to be baked into interpreter
SET(PYTHON_SOURCE_DIR "${INTERPRETER_DIR}/cpython")
SET(PYTHON_MODULE_DIR "${PYTHON_SOURCE_DIR}/build/temp.linux-x86_64-3.7/${PYTHON_SOURCE_DIR}/Modules")
include("CMakePythonModules.txt")
SET(TORCH_PYTHON_DIR "${CMAKE_CURRENT_BINARY_DIR}/../../caffe2/torch/CMakeFiles/torch_python.dir")
include("CMakeTorchPython.txt")
SET_SOURCE_FILES_PROPERTIES(
  ${TORCH_OBJS}
  # ${CMAKE_CURRENT_BINARY_DIR}/../../c10/CMakeFiles/c10.dir/core/TensorImpl.cpp.o
  PROPERTIES
  EXTERNAL_OBJECT true
  GENERATED true
)

# Build cpython
SET(PYTHON_LIB_DIR "${PYTHON_SOURCE_DIR}/lib")
SET(PYTHON_LIB "${PYTHON_LIB_DIR}/libpython3.7m.a")
add_custom_command(
   OUTPUT ${PYTHON_MODULES} ${PYTHON_LIB}
   COMMAND CFLAGS=-fPIC CPPFLAGS=-fPIC ./configure --prefix ${PYTHON_SOURCE_DIR}
   COMMAND CFLAGS=-fPIC CPPFLAGS=-fPIC make -j8
   COMMAND make install
   WORKING_DIRECTORY ${PYTHON_SOURCE_DIR}
   VERBATIM
)

# TODO install pip packages needed by torch in this python
# typing_extensions

# Generate a header that lets interpreter init set up python sys path 
# to point to the cpython built above
SET(PYTORCH_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../")
SET(PY_PATH_STRING "import sys; sys.path = ['', '${PYTHON_LIB_DIR}/python37.zip', '${PYTHON_LIB_DIR}/python3.7', '${PYTHON_LIB_DIR}/python3.7/lib-dynload', '${PYTHON_LIB_DIR}/python3.7/site-packages', '${PYTORCH_ROOT}']")
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/python_path.h
    COMMAND echo "#define PY_PATH_STRING \"${PY_PATH_STRING}\"" > ${CMAKE_CURRENT_SOURCE_DIR}/python_path.h
    VERBATIM
)

# Build the interpreter lib, designed to be standalone and dlopened
# We bake the python and torch_python binding objs into libinterpreter
set(INTERPRETER_LIB_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/python_path.h
  ${INTERPRETER_DIR}/interpreter.cpp
  ${PYTHON_MODULES}
  ${TORCH_OBJS}
  # ${CMAKE_CURRENT_BINARY_DIR}/../../c10/CMakeFiles/c10.dir/core/TensorImpl.cpp.o
)
add_library(interpreter
  ${INTERPRETER_LIB_SOURCES})
target_include_directories(interpreter PRIVATE ${INTERPRETER_DIR})
target_link_libraries(interpreter PRIVATE -L${PYTHON_LIB_DIR} libpython3.7m.a)
target_link_libraries(interpreter PRIVATE crypt crypto ssl pthread dl util m ffi lzma readline nsl ncursesw panelw) # for python builtins
# target_link_libraries(interpreter PRIVATE -Wl,--static c10d)

# handy to have a standalone app to verify linkage and usage of interpreter before embedding it in another lib
set(INTERPRETER_TEST_SOURCES
  ${INTERPRETER_DIR}/test_main.cpp
)
add_executable(interpreter_test ${INTERPRETER_TEST_SOURCES})
target_include_directories(interpreter_test PRIVATE ${INTERPRETER_DIR})
target_link_libraries(interpreter_test PUBLIC gtest dl shm torch)
target_link_libraries(interpreter_test PUBLIC protobuf::libprotobuf-lite)
