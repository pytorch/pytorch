set(LIBRARY_NAME torch_bindings)

file(GLOB_RECURSE SOURCE_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)

add_library(${LIBRARY_NAME} SHARED ${SOURCE_FILES})

target_link_libraries(${LIBRARY_NAME} PRIVATE torch_python_library torch_openreg)

if(WIN32)
        find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
        target_link_libraries(${LIBRARY_NAME} PRIVATE ${Python3_LIBRARIES})
endif()

target_link_directories(${LIBRARY_NAME} PRIVATE ${PYTORCH_INSTALL_DIR}/lib)

if(APPLE)
  set_target_properties(${LIBRARY_NAME} PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
endif()

install(TARGETS ${LIBRARY_NAME}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
