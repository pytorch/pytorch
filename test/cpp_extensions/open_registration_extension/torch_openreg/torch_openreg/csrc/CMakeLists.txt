file(GLOB_RECURSE SOURCE_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)

add_library(${PYTHON_OPENREG_LIB} SHARED ${SOURCE_FILES})
target_link_libraries(${PYTHON_OPENREG_LIB} PRIVATE torch_python torch_openreg)
target_link_directories(${PYTHON_OPENREG_LIB} PRIVATE ${PYTORCH_INSTALL_DIR}/lib)
target_link_options(${PYTHON_OPENREG_LIB} PRIVATE "-Wl,--no-as-needed")
install(TARGETS ${PYTHON_OPENREG_LIB} LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
