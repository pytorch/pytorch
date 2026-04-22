set(LIBRARY_NAME torch_openreg)

file(GLOB_RECURSE SOURCE_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)
list(FILTER SOURCE_FILES EXCLUDE REGEX ".*/tests/.*")

add_library(${LIBRARY_NAME} SHARED ${SOURCE_FILES})

target_link_libraries(${LIBRARY_NAME} PRIVATE torch_cpu_library openreg)
target_include_directories(${LIBRARY_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

install(TARGETS ${LIBRARY_NAME}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

# OpenReg tracer unit tests
if(USE_TEST AND TARGET gtest_main)
  add_executable(test_openreg_tracer
    ${CMAKE_CURRENT_SOURCE_DIR}/profiler/tests/test_openreg_tracer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/profiler/OpenRegTracer.cpp
  )
  target_link_libraries(test_openreg_tracer PRIVATE gtest_main)
  target_include_directories(test_openreg_tracer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
endif()
