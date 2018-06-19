set(LIB_SOURCES
        TensorTypeId.cpp
        TensorTypeIdRegistration.cpp
)

set(TEST_SOURCES
        dummy_test.cpp
)

add_library(dispatch OBJECT ${LIB_SOURCES})
target_enable_style_warnings(dispatch)
target_include_directories(dispatch SYSTEM PUBLIC ${PROJECT_SOURCE_DIR}/third_party/flat_hash_map)

if(BUILD_TEST)
    add_executable(dispatch_test ${TEST_SOURCES} $<TARGET_OBJECTS:dispatch>)
    add_test(NAME dispatch_test COMMAND $<TARGET_FILE:dispatch_test>)
    target_enable_style_warnings(dispatch_test)
    target_link_libraries(dispatch_test gtest_main)
    if(INSTALL_TEST)
        install(TARGETS dispatch_test DESTINATION test)
    endif()
endif()
