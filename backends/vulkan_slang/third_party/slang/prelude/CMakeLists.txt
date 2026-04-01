#
# Use the slang-embed tool to generate a source file for each prelude header.
# Construct a library called 'prelude' to be linked with by slang
#

glob_append(prelude_headers "*-prelude.h")

set(prelude_source)
foreach(input ${prelude_headers})
    get_filename_component(input_name ${input} NAME)
    set(output "${CMAKE_CURRENT_BINARY_DIR}/${input_name}.cpp")
    add_custom_command(
        OUTPUT ${output}
        COMMAND
            slang-embed "${input}" ${output}
            "-I$<JOIN:$<TARGET_PROPERTY:core,INCLUDE_DIRECTORIES>, -I>"
        DEPENDS ${input} slang-embed
        VERBATIM
    )
    list(APPEND prelude_source ${output})
endforeach()

slang_add_target(
    .
    OBJECT
    EXPORT_MACRO_PREFIX SLANG
    EXPLICIT_SOURCE ${prelude_source}
    EXCLUDE_FROM_ALL
    TARGET_NAME prelude
    INCLUDE_DIRECTORIES_PUBLIC
        ${CMAKE_CURRENT_LIST_DIR}
        ${CMAKE_CURRENT_LIST_DIR}/../include
    EXPORT_TYPE_AS ${SLANG_LIB_TYPE}
    LINK_WITH_PRIVATE unordered_dense::unordered_dense
    PUBLIC_HEADERS ${CMAKE_CURRENT_LIST_DIR}/slang*.h
    # It's an object library, so the install step only installs the headers
    INSTALL
    FOLDER generated
)
