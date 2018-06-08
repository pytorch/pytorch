set(LIB_SOURCES
        dummy.cpp
)

set(TEST_SOURCES
        dummy_test.cpp
)

add_library(c10_dispatch OBJECT ${LIB_SOURCES})
target_enable_style_warnings(c10_dispatch)

if(BUILD_TEST)
    add_executable(c10_dispatch_test ${TEST_SOURCES} $<TARGET_OBJECTS:c10_dispatch>)
    add_test(NAME c10_dispatch_test COMMAND $<TARGET_FILE:c10_dispatch_test>)
    target_enable_style_warnings(c10_dispatch_test)
    target_link_libraries(c10_dispatch_test gtest_main)
    if(INSTALL_TEST)
        install(TARGETS c10_dispatch_test DESTINATION test)
    endif()
endif()
