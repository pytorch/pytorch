#
# Generate an embeddable glsl module
#

# This header is generated in the same call to slang-bootstrap that generates
# the core module embed, which can be found in
# ../slang-core-module/CMakeLists.txt
# The target to look for is generate_core_module_headers

# CMake doesn't allow creating an implicit dependency across directory scopes,
# see https://cmake.org/cmake/help/v3.31/prop_sf/GENERATED.html. Because of
# this we add the explicit REQUIRES
slang_add_target(
    .
    MODULE
    FOLDER generated
    LINK_WITH_PRIVATE core
    USE_EXTRA_WARNINGS
    EXPLICIT_SOURCE
        ./slang-embedded-glsl-module.cpp
        ${glsl_module_generated_header}
    REQUIRES generate_core_module_headers
    EXCLUDE_FROM_ALL
    EXTRA_COMPILE_DEFINITIONS_PRIVATE SLANG_SHARED_LIBRARY_TOOL
    EXPORT_SET_NAME SlangTargets
    INCLUDE_DIRECTORIES_PRIVATE ${glsl_module_generated_header_dir}
    INSTALL
)
