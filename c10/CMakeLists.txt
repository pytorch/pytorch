project(c10 CXX C)

set(LIB_SOURCES
        dummy.cpp
)

set(TEST_SOURCES
        dummy_test.cpp
)

if(MSVC)
  # TODO Also add some warning options that MSVC can understand
  set(WARNING_OPTIONS "")
else()
  set(WARNING_OPTIONS
        -Wall
        -Wextra
        -Wold-style-cast
        -Wno-missing-braces
        -Wcast-align
        -Wcast-qual
        -Wctor-dtor-privacy
        -Wdisabled-optimization
        -Wformat=2
        -Winit-self
        -Wmissing-include-dirs
        -Woverloaded-virtual
        -Wredundant-decls
        -Wshadow
        -Wsign-promo
        -Wstrict-overflow=5
        -fdiagnostics-show-option
        -Wconversion
        -Wpedantic
        -Wno-gnu-zero-variadic-macro-arguments
        -Wundef
        -Werror
        )
endif()

add_library(${PROJECT_NAME} OBJECT ${LIB_SOURCES})
target_compile_options(${PROJECT_NAME} PRIVATE ${WARNING_OPTIONS})

if(BUILD_TEST)
    add_executable(${PROJECT_NAME}_test ${TEST_SOURCES} $<TARGET_OBJECTS:${PROJECT_NAME}>)
    add_test(NAME ${PROJECT_NAME}_test COMMAND $<TARGET_FILE:${PROJECT_NAME}_test>)
    target_compile_options(${PROJECT_NAME}_test PRIVATE ${WARNING_OPTIONS})
    target_link_libraries(${PROJECT_NAME}_test gtest_main)
    install(TARGETS ${PROJECT_NAME}_test DESTINATION test)
endif()
