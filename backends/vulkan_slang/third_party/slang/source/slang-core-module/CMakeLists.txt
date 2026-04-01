#
# In this file, for core module source embed and core module embed itself, we define two
# targets, one which includes the embed and one which does not, these are
# linked in as needed to libslang and libslang-without-embedded-core-module (for
# slang-bootstrap)
#
# If MSVC supported weak linking we could just have a single library for each
# and just not link it if we didn't want the embed...
#

#
# Compiling the meta.slang files
#

# List of *.meta.slang headers
set(core_module_meta_source_dir "${slang_SOURCE_DIR}/source/slang")
glob_append(
    core_module_meta_source
    "${core_module_meta_source_dir}/*.meta.slang"
)

set(core_module_meta_output_dir "${CMAKE_CURRENT_BINARY_DIR}/core-module-meta")

# Generate the output file list
set(core_module_meta_generated_headers)
foreach(meta_source ${core_module_meta_source})
    file(
        RELATIVE_PATH
        meta_source_relative
        "${core_module_meta_source_dir}"
        ${meta_source}
    )
    list(
        APPEND
        core_module_meta_generated_headers
        "${core_module_meta_output_dir}/${meta_source_relative}.h"
    )
endforeach()

add_custom_command(
    OUTPUT ${core_module_meta_generated_headers}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${core_module_meta_output_dir}
    COMMAND
        slang-generate ${core_module_meta_source} --target-directory
        ${core_module_meta_output_dir}
    DEPENDS ${core_module_meta_source} slang-generate
    WORKING_DIRECTORY "${core_module_meta_source_dir}"
    VERBATIM
)

#
# Generate embedded core module source library (or a dummy library with just nullptrs)
#

set(core_module_source_common_args
    .
    OBJECT
    EXCLUDE_FROM_ALL
    USE_EXTRA_WARNINGS
    LINK_WITH_PRIVATE
    core
    slang-capability-defs
    slang-fiddle-output
    SPIRV-Headers
    INCLUDE_DIRECTORIES_PRIVATE
    ../slang
    ${core_module_meta_output_dir}
    FOLDER
    generated
    EXPLICIT_SOURCE
    ./slang-embedded-core-module-source.cpp
    EXTRA_COMPILE_DEFINITIONS_PRIVATE
    SLANG_EMBED_CORE_MODULE_SOURCE
    EXPORT_MACRO_PREFIX
    SLANG
    EXPORT_TYPE_AS
    ${SLANG_LIB_TYPE}
)

slang_add_target(
    ${core_module_source_common_args}
    TARGET_NAME slang-embedded-core-module-source
    EXTRA_COMPILE_DEFINITIONS_PRIVATE SLANG_EMBED_CORE_MODULE_SOURCE
    EXPLICIT_SOURCE ${core_module_meta_generated_headers}
    EXCLUDE_FROM_ALL
)

slang_add_target(
    ${core_module_source_common_args}
    TARGET_NAME slang-no-embedded-core-module-source
    EXCLUDE_FROM_ALL
)

#
# Use that source to generate an embeddable core module, this uses
# slang-bootstrap which will include the embedded source we have above
#

set(core_module_generated_header_dir ${CMAKE_CURRENT_BINARY_DIR})
set(core_module_generated_header
    ${core_module_generated_header_dir}/slang-core-module-generated.h
)
set(glsl_module_generated_header_dir
    ${CMAKE_CURRENT_BINARY_DIR}/../slang-glsl-module
)
set(glsl_module_generated_header
    ${glsl_module_generated_header_dir}/slang-glsl-module-generated.h
)

# Propagate to parent directory scope, so they're visible to
# slang-glsl-module/CMakeLists.txt
set(glsl_module_generated_header_dir
    ${glsl_module_generated_header_dir}
    PARENT_SCOPE
)
set(glsl_module_generated_header ${glsl_module_generated_header} PARENT_SCOPE)

add_custom_command(
    OUTPUT ${core_module_generated_header} ${glsl_module_generated_header}
    COMMAND
        slang-bootstrap -archive-type riff-lz4 -save-core-module-bin-source
        ${core_module_generated_header} -save-glsl-module-bin-source
        ${glsl_module_generated_header}
    DEPENDS slang-bootstrap slang-without-embedded-core-module
    VERBATIM
)
# Add a target so that we can depend on the above step when we create the glsl
# module
add_custom_target(
    generate_core_module_headers
    DEPENDS ${core_module_generated_header} ${glsl_module_generated_header}
)
set_target_properties(generate_core_module_headers PROPERTIES FOLDER generated)

set(core_module_common_args
    .
    OBJECT
    EXCLUDE_FROM_ALL
    EXPORT_MACRO_PREFIX
    SLANG
    EXPORT_TYPE_AS
    ${SLANG_LIB_TYPE}
    FOLDER
    generated
    LINK_WITH_PRIVATE
    core
    USE_EXTRA_WARNINGS
    EXPLICIT_SOURCE
    ./slang-embedded-core-module.cpp
)

slang_add_target(
    ${core_module_common_args}
    TARGET_NAME slang-embedded-core-module
    REQUIRES generate_core_module_headers
    EXTRA_COMPILE_DEFINITIONS_PRIVATE SLANG_EMBED_CORE_MODULE
    INCLUDE_DIRECTORIES_PRIVATE ${core_module_generated_header_dir}
)

slang_add_target(
    ${core_module_common_args}
    TARGET_NAME slang-no-embedded-core-module
)
