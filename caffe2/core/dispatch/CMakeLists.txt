set(LIB_SOURCES
        DeviceId.cpp
        Dispatcher.cpp
        DispatchKey.cpp
        DispatchTable.cpp
        KernelRegistration.cpp
        LayoutId.cpp
        OpSchema.cpp
        OpSchemaRegistration.cpp
)

set(TEST_SOURCES
        OpSchema_test.cpp
)

add_library(dispatch OBJECT ${LIB_SOURCES})
target_enable_style_warnings(dispatch)
add_dependencies(dispatch Caffe2_PROTO)

if(BUILD_TEST)
    add_executable(dispatch_test ${TEST_SOURCES} $<TARGET_OBJECTS:dispatch>)
    add_test(NAME dispatch_test COMMAND $<TARGET_FILE:dispatch_test>)
    target_enable_style_warnings(dispatch_test)
    target_link_libraries(dispatch_test gtest_main)
    if(INSTALL_TEST)
        install(TARGETS dispatch_test DESTINATION test)
    endif()
endif()
