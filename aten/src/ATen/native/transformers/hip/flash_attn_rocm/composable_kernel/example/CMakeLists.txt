include_directories(BEFORE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/library/include
)

add_custom_target(examples)

function(add_example_executable EXAMPLE_NAME FILE_NAME)
    message("adding example ${EXAMPLE_NAME}")
    add_executable(${EXAMPLE_NAME} ${FILE_NAME})
    target_link_libraries(${EXAMPLE_NAME} PRIVATE utility)
    add_test(NAME ${EXAMPLE_NAME} COMMAND $<TARGET_FILE:${EXAMPLE_NAME}> ${ARGN})
    add_dependencies(examples ${EXAMPLE_NAME})
    add_dependencies(check ${EXAMPLE_NAME})
    rocm_install(TARGETS ${EXAMPLE_NAME} COMPONENT examples)
endfunction(add_example_executable EXAMPLE_NAME)

function(add_example_executable_no_testing EXAMPLE_NAME FILE_NAME)
    message("adding example ${EXAMPLE_NAME}")
    add_executable(${EXAMPLE_NAME} ${FILE_NAME})
    target_link_libraries(${EXAMPLE_NAME} PRIVATE utility)
    add_dependencies(examples ${EXAMPLE_NAME})
    rocm_install(TARGETS ${EXAMPLE_NAME} COMPONENT examples)
endfunction(add_example_executable_no_testing EXAMPLE_NAME)

# add all example subdir
file(GLOB dir_list LIST_DIRECTORIES true *)
FOREACH(subdir ${dir_list})
    IF(IS_DIRECTORY "${subdir}")
        add_subdirectory(${subdir})
    ENDIF()
ENDFOREACH()
