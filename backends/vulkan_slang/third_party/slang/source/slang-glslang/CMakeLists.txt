#
# Our wrapper for glslang 
#
if(SLANG_ENABLE_SLANG_GLSLANG)
    slang_add_target(
        .
        MODULE
        USE_FEWER_WARNINGS
        LINK_WITH_PRIVATE glslang SPIRV SPIRV-Tools-opt SPIRV-Tools-link
        INCLUDE_DIRECTORIES_PRIVATE ${slang_SOURCE_DIR}/include
        INSTALL
        EXPORT_SET_NAME SlangTargets
        DEBUG_INFO_INSTALL_COMPONENT slang-glslang-debug-info
    )
    # Our only interface is through what we define in source/slang-glslang, in the
    # interests of hygiene, hide anything else we link in.
    add_supported_cxx_linker_flags(
        slang-glslang
        PRIVATE
        "-Wl,--exclude-libs,ALL"
    )
endif()
