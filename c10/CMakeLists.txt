project(c10 CXX C)

include(../cmake/Utils.cmake)

set(LIB_SOURCES
        dummy.cpp
)

set(TEST_SOURCES
        dummy_test.cpp
)

add_library(${PROJECT_NAME} OBJECT ${LIB_SOURCES})
target_enable_style_warnings(${PROJECT_NAME} TRUE)

if(BUILD_TEST)
    add_executable(${PROJECT_NAME}_test ${TEST_SOURCES} $<TARGET_OBJECTS:${PROJECT_NAME}>)
    add_test(NAME ${PROJECT_NAME}_test COMMAND $<TARGET_FILE:${PROJECT_NAME}_test>)
    target_enable_style_warnings(${PROJECT_NAME}_test TRUE)
    target_link_libraries(${PROJECT_NAME}_test gtest_main)
    install(TARGETS ${PROJECT_NAME}_test DESTINATION test)
endif()
