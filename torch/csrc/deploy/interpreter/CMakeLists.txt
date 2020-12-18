SET(INTERPRETER_DIR "${DEPLOY_DIR}/interpreter" )
SET(INTERPRETER_DIR "${DEPLOY_DIR}/interpreter" PARENT_SCOPE)

# Define verbose lists of objects to be baked into interpreter
SET(PYTHON_SOURCE_DIR "${INTERPRETER_DIR}/cpython")
SET(PYTHON_MODULE_DIR "${PYTHON_SOURCE_DIR}/build/temp.linux-x86_64-3.8/${PYTHON_SOURCE_DIR}/Modules")
SET(PYTORCH_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../../")

include("CMakePythonModules.txt")
SET(TORCH_PYTHON_DIR "${CMAKE_CURRENT_BINARY_DIR}/../../../CMakeFiles/torch_python.dir")

add_library(torch_python_static STATIC $<TARGET_OBJECTS:torch_python_obj>)

# Clone cpython
add_custom_command(
   OUTPUT ${PYTHON_SOURCE_DIR}
   COMMENT "Pulling cpython for embedded interpreter, ensure https connectivity."
   COMMAND wget --quiet https://github.com/python/cpython/archive/v3.8.6.tar.gz && tar xf v3.8.6.tar.gz && mv cpython-3.8.6 cpython
   WORKING_DIRECTORY ${INTERPRETER_DIR}
   VERBATIM
)

# Clone typing
SET(TYPING_PKG "${INTERPRETER_DIR}/third_party/typing_extensions.py")
ExternalProject_Add(
  typing
  PREFIX typing
  GIT_REPOSITORY https://github.com/python/typing.git
  GIT_TAG 3.7.4.3
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND cp ../typing/typing_extensions/src_py3/typing_extensions.py ${TYPING_PKG}
  BYPRODUCTS ${TYPING_PKG}
  LOG_OUTPUT_ON_FAILURE True
)

# Build cpython
SET(PYTHON_LIB_DIR "${PYTHON_SOURCE_DIR}/lib")
SET(PYTHON_LIB "${PYTHON_LIB_DIR}/libpython3.8.a")
SET(PYTHON_BIN "${PYTHON_SOURCE_DIR}/python")
add_custom_command(
   OUTPUT ${PYTHON_MODULES} ${PYTHON_LIB} ${PYTHON_BIN}
   COMMAND CFLAGS=-fPIC CPPFLAGS=-fPIC ./configure --prefix ${PYTHON_SOURCE_DIR}
   COMMAND CFLAGS=-fPIC CPPFLAGS=-fPIC make -j8
   COMMAND make install
   WORKING_DIRECTORY ${PYTHON_SOURCE_DIR}
   DEPENDS ${PYTHON_SOURCE_DIR}
   VERBATIM
)

# Freeze the Python source code so we can bundle it with libinterpreter.
SET(FROZEN_DIR "${INTERPRETER_DIR}/frozen")
set(FROZEN_FILES
  ${FROZEN_DIR}/main.c
  ${FROZEN_DIR}/bytecode_0.c
  ${FROZEN_DIR}/bytecode_1.c
  ${FROZEN_DIR}/bytecode_2.c
  ${FROZEN_DIR}/bytecode_3.c
  ${FROZEN_DIR}/bytecode_4.c
)

# Python standard library
file(GLOB PACKAGES_TO_FREEZE ${PYTHON_SOURCE_DIR}/Lib/*)
# torch
list(APPEND PACKAGES_TO_FREEZE ${PYTORCH_ROOT}/torch)
# typing_extensions (needed by torch)
list(APPEND PACKAGES_TO_FREEZE ${TYPING_PKG})

add_custom_command(
   OUTPUT ${FROZEN_FILES}
   WORKING_DIRECTORY ${INTERPRETER_DIR}
   COMMAND mkdir -p ${FROZEN_DIR}
   COMMAND ${PYTHON_BIN} freeze.py ${PACKAGES_TO_FREEZE} --oss --install_dir ${FROZEN_DIR} --verbose
   DEPENDS freeze.py ${PYTHON_BIN} ${PACKAGES_TO_FREEZE} ${TYPING_PKG}
   VERBATIM
)

# Build the interpreter lib, designed to be standalone and dlopened
# We bake the python and torch_python binding objs into libinterpreter
set(LINKER_SCRIPT "${INTERPRETER_DIR}/hide_symbols.script")
set(INTERPRETER_LIB_SOURCES
  ${INTERPRETER_DIR}/interpreter.cpp
  ${FROZEN_FILES}
  ${PYTHON_MODULES}
  ${LINKER_SCRIPT}
)
add_library(interpreter
  ${INTERPRETER_LIB_SOURCES} ${LINKER_SCRIPT})
set_property(TARGET interpreter APPEND_STRING PROPERTY
             LINK_FLAGS " -Wl,--version-script=${LINKER_SCRIPT}")
target_compile_options(
    interpreter PRIVATE
    -fvisibility=hidden
)
target_include_directories(interpreter PRIVATE ${INTERPRETER_DIR})
target_link_libraries(interpreter PRIVATE -L${PYTHON_LIB_DIR} libpython3.8.a)
target_link_libraries(interpreter PRIVATE crypt crypto ssl pthread dl util m z ffi lzma readline nsl ncursesw panelw) # for python builtins
# target_link_libraries(interpreter PRIVATE -Wl,--static c10d)
target_link_libraries(interpreter PRIVATE fmt::fmt-header-only)
target_link_libraries(interpreter PRIVATE torch_python_static)
target_link_libraries(interpreter PRIVATE protobuf::libprotobuf-lite)

# handy to have a standalone app to verify linkage and usage of interpreter before embedding it in another lib
set(INTERPRETER_TEST_SOURCES
  ${INTERPRETER_DIR}/test_main.cpp
)
add_executable(interpreter_test ${INTERPRETER_TEST_SOURCES})
target_include_directories(interpreter_test PRIVATE ${INTERPRETER_DIR})
target_link_libraries(interpreter_test PUBLIC gtest dl shm torch)
target_link_libraries(interpreter_test PUBLIC protobuf::libprotobuf-lite)

set(INTERPRETER_APP_SOURCES
  ${INTERPRETER_DIR}/app_main.cpp
)
#add_executable(interpreter_app ${INTERPRETER_APP_SOURCES})
#target_include_directories(interpreter_app PRIVATE ${INTERPRETER_DIR})
#target_link_libraries(interpreter_app PUBLIC dl shm torch)
#target_link_libraries(interpreter_app PUBLIC protobuf::libprotobuf-lite)
