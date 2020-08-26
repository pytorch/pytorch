SET(INTERPRETER_DIR "${TORCHPY_DIR}/interpreter" )
SET(INTERPRETER_DIR "${TORCHPY_DIR}/interpreter" PARENT_SCOPE)
set(INTERPRETER_LIB_SOURCES
  ${INTERPRETER_DIR}/interpreter.cpp
)
add_library(interpreter ${INTERPRETER_LIB_SOURCES})
target_include_directories(interpreter PRIVATE ${INTERPRETER_DIR})
target_link_libraries(interpreter PRIVATE torch torch_python -Wl,-rpath,/home/whc/local/pytorch-env/lib/ ${PYTHON_LIBRARIES})
target_link_libraries(interpreter PRIVATE dl)