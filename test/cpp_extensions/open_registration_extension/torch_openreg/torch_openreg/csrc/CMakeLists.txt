set(LIBRARY_NAME torch_bindings)

file(GLOB_RECURSE SOURCE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

add_library(${LIBRARY_NAME} SHARED ${SOURCE_FILES})

target_link_libraries(${LIBRARY_NAME} PRIVATE torch_python torch_openreg)
target_link_directories(${LIBRARY_NAME} PRIVATE ${PYTORCH_INSTALL_DIR}/lib)

install(TARGETS ${LIBRARY_NAME} LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
