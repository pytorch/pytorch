set(LIBRARY_NAME torch_bindings)

file(GLOB_RECURSE SOURCE_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)

add_library(${LIBRARY_NAME} SHARED ${SOURCE_FILES})

if(WIN32)
    # For Windows, we need specific PyTorch libraries
    find_library(TORCH_PYTHON_LIBRARY torch_python PATHS ${PYTORCH_INSTALL_DIR}/lib REQUIRED)
    find_library(TORCH_CPU_LIBRARY torch_cpu PATHS ${PYTORCH_INSTALL_DIR}/lib REQUIRED)
    find_library(C10_LIBRARY c10 PATHS ${PYTORCH_INSTALL_DIR}/lib REQUIRED)
    
    target_link_libraries(${LIBRARY_NAME} PRIVATE 
        ${TORCH_PYTHON_LIBRARY}
        ${TORCH_CPU_LIBRARY}
        ${C10_LIBRARY}
        torch_openreg
    )
else()
    target_link_libraries(${LIBRARY_NAME} PRIVATE torch_python torch_openreg)
endif()

if(WIN32)
    if(DEFINED PYTHON_LIBRARY_DIR AND DEFINED PYTHON_LIBRARY)
        target_link_directories(${LIBRARY_NAME} PRIVATE ${PYTHON_LIBRARY_DIR})
        target_link_libraries(${LIBRARY_NAME} PRIVATE ${PYTHON_LIBRARY})
    else()
        find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
        target_link_libraries(${LIBRARY_NAME} PRIVATE ${Python3_LIBRARIES})
    endif()
endif()

if(WIN32)
    set_target_properties(${LIBRARY_NAME} PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

target_link_directories(${LIBRARY_NAME} PRIVATE ${PYTORCH_INSTALL_DIR}/lib)
target_include_directories(${LIBRARY_NAME} PRIVATE ${PYTORCH_INSTALL_DIR}/include)

if(APPLE)
  set_target_properties(${LIBRARY_NAME} PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
endif()

if(WIN32)
    install(TARGETS ${LIBRARY_NAME} 
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )
else()
    install(TARGETS ${LIBRARY_NAME} LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()
