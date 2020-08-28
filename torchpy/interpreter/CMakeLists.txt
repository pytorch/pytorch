SET(INTERPRETER_DIR "${TORCHPY_DIR}/interpreter" )
SET(INTERPRETER_DIR "${TORCHPY_DIR}/interpreter" PARENT_SCOPE)

SET(PYTHON_SOURCE_DIR "${INTERPRETER_DIR}/cpython")
SET(PYTHON_MODULE_DIR "${PYTHON_SOURCE_DIR}/build/temp.linux-x86_64-3.7/${PYTHON_SOURCE_DIR}/Modules")
SET(PYTHON_MODULES
  ${PYTHON_MODULE_DIR}/arraymodule.o
  ${PYTHON_MODULE_DIR}/_asynciomodule.o
  ${PYTHON_MODULE_DIR}/audioop.o
  ${PYTHON_MODULE_DIR}/binascii.o
  ${PYTHON_MODULE_DIR}/_bisectmodule.o
  ${PYTHON_MODULE_DIR}/_blake2/blake2module.o ${PYTHON_MODULE_DIR}/_blake2/blake2b_impl.o ${PYTHON_MODULE_DIR}/_blake2/blake2s_impl.o
  ${PYTHON_MODULE_DIR}/_bz2module.o
  ${PYTHON_MODULE_DIR}/cmathmodule.o
  # ${PYTHON_MODULE_DIR}/_math.o
  ${PYTHON_MODULE_DIR}/cjkcodecs/_codecs_cn.o
  ${PYTHON_MODULE_DIR}/cjkcodecs/_codecs_hk.o
  ${PYTHON_MODULE_DIR}/cjkcodecs/_codecs_iso2022.o
  ${PYTHON_MODULE_DIR}/cjkcodecs/_codecs_jp.o
  ${PYTHON_MODULE_DIR}/cjkcodecs/_codecs_kr.o
  ${PYTHON_MODULE_DIR}/cjkcodecs/_codecs_tw.o
  ${PYTHON_MODULE_DIR}/_contextvarsmodule.o
  ${PYTHON_MODULE_DIR}/_cryptmodule.o
  ${PYTHON_MODULE_DIR}/_csv.o
  ${PYTHON_MODULE_DIR}/_ctypes/_ctypes.o ${PYTHON_MODULE_DIR}/_ctypes/callbacks.o ${PYTHON_MODULE_DIR}/_ctypes/callproc.o ${PYTHON_MODULE_DIR}/_ctypes/stgdict.o ${PYTHON_MODULE_DIR}/_ctypes/cfield.o
  ${PYTHON_MODULE_DIR}/_ctypes/_ctypes_test.o
  ${PYTHON_MODULE_DIR}/_cursesmodule.o
  ${PYTHON_MODULE_DIR}/_curses_panel.o
  ${PYTHON_MODULE_DIR}/_datetimemodule.o
  ${PYTHON_MODULE_DIR}/_decimal/_decimal.o ${PYTHON_MODULE_DIR}/_decimal/libmpdec/basearith.o ${PYTHON_MODULE_DIR}/_decimal/libmpdec/constants.o ${PYTHON_MODULE_DIR}/_decimal/libmpdec/context.o ${PYTHON_MODULE_DIR}/_decimal/libmpdec/convolute.o ${PYTHON_MODULE_DIR}/_decimal/libmpdec/crt.o ${PYTHON_MODULE_DIR}/_decimal/libmpdec/difradix2.o ${PYTHON_MODULE_DIR}/_decimal/libmpdec/fnt.o ${PYTHON_MODULE_DIR}/_decimal/libmpdec/fourstep.o ${PYTHON_MODULE_DIR}/_decimal/libmpdec/io.o ${PYTHON_MODULE_DIR}/_decimal/libmpdec/memory.o ${PYTHON_MODULE_DIR}/_decimal/libmpdec/mpdecimal.o ${PYTHON_MODULE_DIR}/_decimal/libmpdec/numbertheory.o ${PYTHON_MODULE_DIR}/_decimal/libmpdec/sixstep.o ${PYTHON_MODULE_DIR}/_decimal/libmpdec/transpose.o
  ${PYTHON_MODULE_DIR}/_elementtree.o
  ${PYTHON_MODULE_DIR}/fcntlmodule.o
  ${PYTHON_MODULE_DIR}/grpmodule.o
  ${PYTHON_MODULE_DIR}/_hashopenssl.o
  ${PYTHON_MODULE_DIR}/_heapqmodule.o
  ${PYTHON_MODULE_DIR}/_json.o
  ${PYTHON_MODULE_DIR}/_lsprof.o
  ${PYTHON_MODULE_DIR}/_lzmamodule.o
  ${PYTHON_MODULE_DIR}/mathmodule.o
  ${PYTHON_MODULE_DIR}/md5module.o
  ${PYTHON_MODULE_DIR}/mmapmodule.o
  ${PYTHON_MODULE_DIR}/cjkcodecs/multibytecodec.o
  ${PYTHON_MODULE_DIR}/_multiprocessing/multiprocessing.o ${PYTHON_MODULE_DIR}/_multiprocessing/semaphore.o
  ${PYTHON_MODULE_DIR}/nismodule.o
  ${PYTHON_MODULE_DIR}/_opcode.o
  ${PYTHON_MODULE_DIR}/ossaudiodev.o
  ${PYTHON_MODULE_DIR}/parsermodule.o
  ${PYTHON_MODULE_DIR}/_pickle.o
  ${PYTHON_MODULE_DIR}/_posixsubprocess.o
  ${PYTHON_MODULE_DIR}/pyexpat.o ${PYTHON_MODULE_DIR}/expat/xmlparse.o ${PYTHON_MODULE_DIR}/expat/xmlrole.o ${PYTHON_MODULE_DIR}/expat/xmltok.o
  ${PYTHON_MODULE_DIR}/_queuemodule.o
  ${PYTHON_MODULE_DIR}/_randommodule.o
  ${PYTHON_MODULE_DIR}/readline.o
  ${PYTHON_MODULE_DIR}/resource.o
  ${PYTHON_MODULE_DIR}/selectmodule.o
  ${PYTHON_MODULE_DIR}/sha1module.o
  ${PYTHON_MODULE_DIR}/sha256module.o
  ${PYTHON_MODULE_DIR}/_sha3/sha3module.o
  ${PYTHON_MODULE_DIR}/sha512module.o
  ${PYTHON_MODULE_DIR}/socketmodule.o
  ${PYTHON_MODULE_DIR}/spwdmodule.o
  ${PYTHON_MODULE_DIR}/_ssl.o
  ${PYTHON_MODULE_DIR}/_struct.o
  ${PYTHON_MODULE_DIR}/syslogmodule.o
  ${PYTHON_MODULE_DIR}/termios.o
  ${PYTHON_MODULE_DIR}/_testbuffer.o
  ${PYTHON_MODULE_DIR}/_testcapimodule.o
  ${PYTHON_MODULE_DIR}/_testimportmultiple.o
  ${PYTHON_MODULE_DIR}/_testmultiphase.o
  ${PYTHON_MODULE_DIR}/unicodedata.o
  ${PYTHON_MODULE_DIR}/xxlimited.o
  ${PYTHON_MODULE_DIR}/_xxtestfuzz/_xxtestfuzz.o ${PYTHON_MODULE_DIR}/_xxtestfuzz/fuzzer.o
  ${PYTHON_MODULE_DIR}/zlibmodule.o
)
SET(PYTHON_LIB_DIR "${PYTHON_SOURCE_DIR}/lib/")
SET(PYTHON_LIB "${PYTHON_LIB_DIR}/libpython3.7m.a")
add_custom_command(
   OUTPUT ${PYTHON_MODULES} ${PYTHON_LIB}
   COMMAND CFLAGS=-fPIC CPPFLAGS=-fPIC ./configure --prefix ${PYTHON_SOURCE_DIR}
   COMMAND CFLAGS=-fPIC CPPFLAGS=-fPIC make -j8
   COMMAND make install
   WORKING_DIRECTORY ${PYTHON_SOURCE_DIR}
   VERBATIM
)

SET(PYTORCH_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../")
SET(PY_PATH_STRING "import sys; sys.path = ['', '${PYTHON_LIB_DIR}/python37.zip', '${PYTHON_LIB_DIR}/python3.7', '${PYTHON_LIB_DIR}/lib-dynload', '${PYTHON_LIB_DIR}/site-packages', '${PYTORCH_ROOT}']")
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/python_path.h
    COMMAND echo "#define PY_PATH_STRING \"${PY_PATH_STRING}\"" > ${CMAKE_CURRENT_SOURCE_DIR}/python_path.h
    VERBATIM
)

set(INTERPRETER_LIB_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/python_path.h
  ${INTERPRETER_DIR}/interpreter.cpp
  ${PYTHON_MODULES}
)
add_library(interpreter
  ${INTERPRETER_LIB_SOURCES})
target_include_directories(interpreter PRIVATE ${INTERPRETER_DIR})

target_link_libraries(interpreter PRIVATE -L${PYTHON_LIB_DIR} libpython3.7m.a)
target_link_libraries(interpreter PRIVATE crypt crypto ssl pthread dl util m ffi lzma readline nsl ncursesw panelw) # for python builtins
target_link_libraries(interpreter PRIVATE torch torch_python)

# handy to have a standalone app to verify linkage and usage of interpreter before embedding it in another lib
set(INTERPRETER_TEST_SOURCES
  ${INTERPRETER_DIR}/test_main.cpp
)
add_executable(interpreter_test ${INTERPRETER_TEST_SOURCES})
target_include_directories(interpreter_test PRIVATE ${INTERPRETER_DIR})
target_link_libraries(interpreter_test PUBLIC gtest dl)
