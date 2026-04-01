#
# A function to make target specification a little more declarative
#
# See the comments on the options below for usage
#
function(slang_add_target dir type)
    set(no_value_args
        # Don't include in the 'all' target
        EXCLUDE_FROM_ALL
        # This is loaded at runtime as a shared library
        SHARED_LIBRARY_TOOL
        # -Wextra
        USE_EXTRA_WARNINGS
        # don't set -Wall
        USE_FEWER_WARNINGS
        # Make this a Windows app, rather than a console app, only makes a
        # difference when compiling for Windows
        WIN32_EXECUTABLE
        # Install this target for a non-component install
        INSTALL
        # Don't include any source in this target, this is a complement to
        # EXPLICIT_SOURCE, and doesn't interact with EXTRA_SOURCE_DIRS
        NO_SOURCE
        # Don't generate split debug info for this target
        NO_SPLIT_DEBUG_INFO
    )
    set(single_value_args
        # Set the target name, useful for multiple targets from the same
        # directory.
        # By default this is the name of the last directory component given
        TARGET_NAME
        # Set the output name, otherwise defaults to the target name
        OUTPUT_NAME
        # Set an explicit output directory relative to the cmake binary
        # directory. otherwise defaults to the binary directory root.
        # Outputs are always placed in a further subdirectory according to
        # build config
        OUTPUT_DIR
        # If this is a shared library then the ${EXPORT_MACRO_PREFIX}_DYNAMIC and
        # ${EXPORT_MACRO_PREFIX}_DYNAMIC_EXPORT macros are set for using and
        # building respectively
        EXPORT_MACRO_PREFIX
        # Ignore target type and use a particular style of export macro
        # _DYNAMIC or _STATIC, this is useful when the target type is OBJECT 
        # pass in STATIC or SHARED
        EXPORT_TYPE_AS
        # The folder in which to place this target for IDE-based generators (VS
        # and XCode)
        FOLDER
        # The working directory for debugging
        DEBUG_DIR
        # Install this target as part of a component
        INSTALL_COMPONENT
        # Override the debug info component name for installation
        # explicit name instead, used for externally built things such as
        # slang-glslang and slang-llvm which have large pdb files
        DEBUG_INFO_INSTALL_COMPONENT
        # The name of the Export set to associate with this installed target
        EXPORT_SET_NAME
    )
    set(multi_value_args
        # Use exactly these sources, instead of globbing from the directory
        # argument
        EXPLICIT_SOURCE
        # Additional directories from which to glob source
        EXTRA_SOURCE_DIRS
        # Additional compile definitions and options
        EXTRA_COMPILE_DEFINITIONS_PRIVATE
        EXTRA_COMPILE_DEFINITIONS_PUBLIC
        EXTRA_COMPILE_OPTIONS_PRIVATE
        # Targets with which to link privately
        LINK_WITH_PRIVATE
        # Targets with which to link publicly, for example if their headers
        # appear in our headers
        LINK_WITH_PUBLIC
        # Frameworks with which to link privately
        LINK_WITH_FRAMEWORK
        # Targets whose headers we use, but don't link with
        INCLUDE_FROM_PRIVATE
        # Targets whose headers we use in our headers, so need to make sure
        # dependencies of this target also include them
        INCLUDE_FROM_PUBLIC
        # Any include directories other targets need to use this target
        INCLUDE_DIRECTORIES_PUBLIC
        # Any include directories this target only needs
        INCLUDE_DIRECTORIES_PRIVATE
        # Add a dependency on the new target to the specified targets
        REQUIRED_BY
        # Add a dependency to the new target on the specified targets
        REQUIRES
        # Add a dependency to the new target on the specified targets if they exist
        OPTIONAL_REQUIRES
        # Globs for any headers to install
        PUBLIC_HEADERS
    )
    cmake_parse_arguments(
        ARG
        "${no_value_args}"
        "${single_value_args}"
        "${multi_value_args}"
        ${ARGN}
    )

    if(DEFINED ARG_UNPARSED_ARGUMENTS OR DEFINED ARG_KEYWORDS_MISSING_VALUES)
        foreach(unparsed_arg ${ARG_UNPARSED_ARGUMENTS})
            message(
                SEND_ERROR
                "Unrecognized argument in slang_add_target: ${unparsed_arg}"
            )
        endforeach()
        foreach(bad_kwarg ${ARG_KEYWORDS_MISSING_VALUES})
            message(
                SEND_ERROR
                "Keyword argument missing values in slang_add_target: ${bad_kwarg}"
            )
        endforeach()
        return()
    endif()

    #
    # Set up some variables, including the target name
    #
    get_filename_component(dir_absolute ${dir} ABSOLUTE)
    if(DEFINED ARG_TARGET_NAME)
        set(target ${ARG_TARGET_NAME})
    else()
        get_filename_component(target ${dir_absolute} NAME)
    endif()

    #
    # Find the source for this target
    #
    if(ARG_NO_SOURCE)
    elseif(ARG_EXPLICIT_SOURCE)
        list(APPEND source ${ARG_EXPLICIT_SOURCE})
    else()
        slang_glob_sources(source ${dir})
    endif()
    foreach(extra_dir ${ARG_EXTRA_SOURCE_DIRS})
        slang_glob_sources(source ${extra_dir})
    endforeach()

    #
    # Create the target
    #
    set(library_types
        STATIC
        SHARED
        OBJECT
        MODULE
        ALIAS
    )
    if(type STREQUAL "EXECUTABLE")
        add_executable(${target} ${source})
    elseif(type STREQUAL "LIBRARY")
        add_library(${target} ${source})
    elseif(type IN_LIST library_types)
        add_library(${target} ${type} ${source})
    else()
        message(
            SEND_ERROR
            "Unsupported target type ${type} in slang_add_target"
        )
        return()
    endif()

    # Enable link-time optimization for release builds
    # See: https://cmake.org/cmake/help/latest/prop_tgt/INTERPROCEDURAL_OPTIMIZATION.html
    if(SLANG_ENABLE_RELEASE_LTO)
        set_target_properties(
            ${target}
            PROPERTIES
                INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE
                INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO TRUE
        )
    endif()

    #
    # Set the output directory
    #
    # We don't want the output directory to be sensitive to where
    # slang_add_target is called from, so set it explicitly here.
    #
    if(DEFINED ARG_OUTPUT_DIR)
        set(output_dir "${CMAKE_BINARY_DIR}/${ARG_OUTPUT_DIR}/$<CONFIG>")
    else()
        # Default to placing things in the cmake binary root.
        #
        # While it would be nice to place things according to their
        # subdirectory, Windows' inflexibility in being able to find DLLs makes
        # this tricky there.
        set(output_dir "${CMAKE_BINARY_DIR}/$<CONFIG>")
    endif()
    set(archive_subdir ${library_subdir})
    if(type STREQUAL "MODULE")
        set(library_subdir ${module_subdir})
    endif()

    # Respect user-defined CMAKE_*_OUTPUT_DIRECTORY variables if they are set
    if(DEFINED CMAKE_ARCHIVE_OUTPUT_DIRECTORY)
        set(archive_output_dir "${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}")
    else()
        set(archive_output_dir "${output_dir}/${archive_subdir}")
    endif()

    if(DEFINED CMAKE_LIBRARY_OUTPUT_DIRECTORY)
        set(library_output_dir "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
    else()
        set(library_output_dir "${output_dir}/${library_subdir}")
    endif()

    if(DEFINED CMAKE_RUNTIME_OUTPUT_DIRECTORY)
        set(runtime_output_dir "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
        set(pdb_output_dir "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
    else()
        set(runtime_output_dir "${output_dir}/${runtime_subdir}")
        set(pdb_output_dir "${output_dir}/${runtime_subdir}")
    endif()

    set_target_properties(
        ${target}
        PROPERTIES
            ARCHIVE_OUTPUT_DIRECTORY "${archive_output_dir}"
            LIBRARY_OUTPUT_DIRECTORY "${library_output_dir}"
            RUNTIME_OUTPUT_DIRECTORY "${runtime_output_dir}"
            PDB_OUTPUT_DIRECTORY "${pdb_output_dir}"
    )

    # For Multi-Config generators we also need to set per-config output directories
    # if user-defined paths are provided
    if(DEFINED CMAKE_ARCHIVE_OUTPUT_DIRECTORY)
        set_target_properties(
            ${target}
            PROPERTIES
                ARCHIVE_OUTPUT_DIRECTORY_DEBUG
                    "${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}"
                ARCHIVE_OUTPUT_DIRECTORY_RELEASE
                    "${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}"
                ARCHIVE_OUTPUT_DIRECTORY_RELWITHDEBINFO
                    "${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}"
                ARCHIVE_OUTPUT_DIRECTORY_MINSIZEREL
                    "${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}"
        )
    endif()

    if(DEFINED CMAKE_LIBRARY_OUTPUT_DIRECTORY)
        set_target_properties(
            ${target}
            PROPERTIES
                LIBRARY_OUTPUT_DIRECTORY_DEBUG
                    "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}"
                LIBRARY_OUTPUT_DIRECTORY_RELEASE
                    "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}"
                LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO
                    "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}"
                LIBRARY_OUTPUT_DIRECTORY_MINSIZEREL
                    "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}"
        )
    endif()

    if(DEFINED CMAKE_RUNTIME_OUTPUT_DIRECTORY)
        set_target_properties(
            ${target}
            PROPERTIES
                RUNTIME_OUTPUT_DIRECTORY_DEBUG
                    "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
                RUNTIME_OUTPUT_DIRECTORY_RELEASE
                    "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
                RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO
                    "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
                RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL
                    "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
                PDB_OUTPUT_DIRECTORY_DEBUG "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
                PDB_OUTPUT_DIRECTORY_RELEASE "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
                PDB_OUTPUT_DIRECTORY_RELWITHDEBINFO
                    "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
                PDB_OUTPUT_DIRECTORY_MINSIZEREL
                    "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
        )
    endif()

    set(debug_configs "Debug,RelWithDebInfo")
    if(SLANG_ENABLE_RELEASE_DEBUG_INFO)
        set(debug_configs "Debug,RelWithDebInfo,Release")
    endif()

    set_target_properties(
        ${target}
        PROPERTIES
            MSVC_DEBUG_INFORMATION_FORMAT
                "$<$<CONFIG:${debug_configs}>:Embedded>"
    )
    if(MSVC)
        target_link_options(
            ${target}
            PRIVATE "$<$<CONFIG:${debug_configs}>:/DEBUG>"
        )
    else()
        target_compile_options(
            ${target}
            PRIVATE "$<$<CONFIG:${debug_configs}>:-g>"
        )
    endif()

    #
    # Set common compile options and properties
    #
    if(ARG_USE_EXTRA_WARNINGS)
        set_default_compile_options(${target} USE_EXTRA_WARNINGS)
    elseif(ARG_USE_FEWER_WARNINGS)
        set_default_compile_options(${target} USE_FEWER_WARNINGS)
    else()
        set_default_compile_options(${target})
    endif()

    # Set debug info options if not disabled
    # Determine if this target produces a binary that can have debug info
    if(
        NOT ARG_NO_SPLIT_DEBUG_INFO
        AND type MATCHES "^(EXECUTABLE|SHARED|MODULE)$"
        AND SLANG_ENABLE_SPLIT_DEBUG_INFO
    )
        set(generate_split_debug_info TRUE)
    else()
        set(generate_split_debug_info FALSE)
    endif()

    if(generate_split_debug_info)
        if(MSVC)
            set_target_properties(
                ${target}
                PROPERTIES
                    COMPILE_PDB_NAME "${target}"
                    COMPILE_PDB_OUTPUT_DIRECTORY "${output_dir}"
            )
        else()
            if(CMAKE_SYSTEM_NAME MATCHES "Darwin")
                # macOS - use dsymutil with --flat to create separate debug file
                add_custom_command(
                    TARGET ${target}
                    POST_BUILD
                    COMMAND
                        dsymutil --flat $<TARGET_FILE:${target}> -o
                        $<TARGET_FILE:${target}>.dwarf
                    COMMAND chmod 644 $<TARGET_FILE:${target}>.dwarf
                    COMMAND ${CMAKE_STRIP} -S $<TARGET_FILE:${target}>
                    WORKING_DIRECTORY ${output_dir}
                    VERBATIM
                )
            else()
                add_custom_command(
                    TARGET ${target}
                    POST_BUILD
                    COMMAND
                        ${CMAKE_OBJCOPY} --only-keep-debug
                        $<TARGET_FILE:${target}> $<TARGET_FILE:${target}>.dwarf
                    WORKING_DIRECTORY ${output_dir}
                    VERBATIM
                )
                # We may be building for Android on a Windows host, where chmod isn't available or needed.
                if(NOT CMAKE_HOST_WIN32)
                    add_custom_command(
                        TARGET ${target}
                        POST_BUILD
                        COMMAND chmod 644 $<TARGET_FILE:${target}>.dwarf
                        WORKING_DIRECTORY ${output_dir}
                        VERBATIM
                    )
                endif()
                add_custom_command(
                    TARGET ${target}
                    POST_BUILD
                    COMMAND
                        ${CMAKE_STRIP} --strip-debug $<TARGET_FILE:${target}>
                    COMMAND
                        ${CMAKE_OBJCOPY}
                        --add-gnu-debuglink=$<TARGET_FILE:${target}>.dwarf
                        $<TARGET_FILE:${target}>
                    WORKING_DIRECTORY ${output_dir}
                    VERBATIM
                )
            endif()
        endif()
    endif()

    set_target_properties(
        ${target}
        PROPERTIES EXCLUDE_FROM_ALL ${ARG_EXCLUDE_FROM_ALL}
    )

    set_target_properties(
        ${target}
        PROPERTIES WIN32_EXECUTABLE ${ARG_WIN32_EXECUTABLE}
    )

    if(DEFINED ARG_OUTPUT_NAME)
        set_target_properties(
            ${target}
            PROPERTIES OUTPUT_NAME ${ARG_OUTPUT_NAME}
        )
    endif()

    if(DEFINED ARG_FOLDER)
        set_target_properties(${target} PROPERTIES FOLDER ${ARG_FOLDER})
    endif()

    if(DEFINED ARG_DEBUG_DIR)
        set_target_properties(
            ${target}
            PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY ${ARG_DEBUG_DIR}
        )
    endif()

    #
    # Link and include from dependencies
    #
    target_link_libraries(${target} PRIVATE ${ARG_LINK_WITH_PRIVATE})
    target_link_libraries(${target} PUBLIC ${ARG_LINK_WITH_PUBLIC})

    if(CMAKE_SYSTEM_NAME MATCHES "Darwin")
        foreach(link_framework ${ARG_LINK_WITH_FRAMEWORK})
            target_link_libraries(
                ${target}
                PRIVATE "-framework ${link_framework}"
            )
        endforeach()
    endif()

    foreach(include_from ${ARG_INCLUDE_FROM_PRIVATE})
        target_include_directories(
            ${target}
            PRIVATE
                $<TARGET_PROPERTY:${include_from},INTERFACE_INCLUDE_DIRECTORIES>
        )
    endforeach()
    foreach(include_from ${ARG_INCLUDE_FROM_PUBLIC})
        target_include_directories(
            ${target}
            PUBLIC
                $<TARGET_PROPERTY:${include_from},INTERFACE_INCLUDE_DIRECTORIES>
        )
    endforeach()

    #
    # Set our exported include directories
    #
    foreach(inc ${ARG_INCLUDE_DIRECTORIES_PUBLIC})
        get_filename_component(inc_abs ${inc} ABSOLUTE)
        target_include_directories(
            ${target}
            PUBLIC "$<BUILD_INTERFACE:${inc_abs}>"
        )
    endforeach()
    foreach(inc ${ARG_INCLUDE_DIRECTORIES_PRIVATE})
        get_filename_component(inc_abs ${inc} ABSOLUTE)
        target_include_directories(
            ${target}
            PRIVATE "$<BUILD_INTERFACE:${inc_abs}>"
        )
    endforeach()

    #
    # Set up export macros
    #
    get_target_property(target_type ${target} TYPE)
    if(DEFINED ARG_EXPORT_MACRO_PREFIX)
        if(
            target_type STREQUAL SHARED_LIBRARY
            OR target_type STREQUAL MODULE_LIBRARY
            OR ARG_EXPORT_TYPE_AS STREQUAL SHARED
            OR ARG_EXPORT_TYPE_AS STREQUAL MODULE
        )
            target_compile_definitions(
                ${target}
                PUBLIC "${ARG_EXPORT_MACRO_PREFIX}_DYNAMIC"
                PRIVATE "${ARG_EXPORT_MACRO_PREFIX}_DYNAMIC_EXPORT"
            )
        elseif(
            target_type STREQUAL STATIC_LIBRARY
            OR ARG_EXPORT_TYPE_AS STREQUAL STATIC
        )
            target_compile_definitions(
                ${target}
                PUBLIC "${ARG_EXPORT_MACRO_PREFIX}_STATIC"
            )
        else()
            message(
                WARNING
                "unhandled case in slang_add_target while setting export macro"
            )
        endif()
    endif()

    #
    # Other dependencies
    #
    foreach(requirer ${ARG_REQUIRED_BY})
        add_dependencies(${requirer} ${target})
    endforeach()

    if(DEFINED ARG_REQUIRES)
        add_dependencies(${target} ${ARG_REQUIRES})
    endif()

    foreach(required ${ARG_OPTIONAL_REQUIRES})
        if(TARGET ${required})
            add_dependencies(${target} ${required})
        endif()
    endforeach()

    #
    # Other preprocessor defines and options
    #
    if(ARG_EXTRA_COMPILE_DEFINITIONS_PRIVATE)
        target_compile_definitions(
            ${target}
            PRIVATE ${ARG_EXTRA_COMPILE_DEFINITIONS_PRIVATE}
        )
    endif()
    if(ARG_EXTRA_COMPILE_DEFINITIONS_PUBLIC)
        target_compile_definitions(
            ${target}
            PUBLIC ${ARG_EXTRA_COMPILE_DEFINITIONS_PUBLIC}
        )
    endif()
    if(ARG_EXTRA_COMPILE_OPTIONS_PRIVATE)
        target_compile_options(
            ${target}
            PRIVATE ${ARG_EXTRA_COMPILE_OPTIONS_PRIVATE}
        )
    endif()

    #
    # Since we do a lot of dynamic loading, unconditionally set the build rpath
    # to find our libraries. Ordinarily CMake would sort this out, but we do
    # have libraries which at build time don't depend on any other shared
    # libraries of ours but which do load them at runtime, hence the need to do
    # this explicitly here.
    #
    if(CMAKE_SYSTEM_NAME MATCHES "Darwin")
        set(ORIGIN "@loader_path")
    else()
        set(ORIGIN "$ORIGIN")
    endif()
    set_property(
        TARGET ${target}
        APPEND
        PROPERTY BUILD_RPATH "${ORIGIN}/../${library_subdir};${ORIGIN}"
    )
    set_property(
        TARGET ${target}
        APPEND
        PROPERTY INSTALL_RPATH "${ORIGIN}/../${library_subdir};${ORIGIN}"
    )

    # On the same topic, give everything a dylib suffix on Mac OS
    if(CMAKE_SYSTEM_NAME MATCHES "Darwin" AND type STREQUAL "MODULE")
        set_property(TARGET ${target} PROPERTY SUFFIX ".dylib")
    endif()

    #
    # Mark headers for installation
    #
    if(ARG_PUBLIC_HEADERS)
        if(NOT ARG_INSTALL)
            message(
                WARNING
                "${target} was declared with PUBLIC_HEADERS but without INSTALL, the former will do nothing"
            )
        endif()

        glob_append(public_headers ${ARG_PUBLIC_HEADERS})
        if(NOT public_headers)
            message(WARNING "${target}'s PUBLIC_HEADER globs found no matches")
        else()
            set_target_properties(
                ${target}
                PROPERTIES PUBLIC_HEADER "${public_headers}"
            )
        endif()
    endif()

    #
    # Mark for installation
    #
    macro(i)
        if(ARG_EXPORT_SET_NAME)
            set(export_args EXPORT ${ARG_EXPORT_SET_NAME})
        else()
            if(type MATCHES "^(EXECUTABLE|SHARED|MODULE)$")
                message(
                    WARNING
                    "Target ${target} is set to be INSTALLED but EXPORT_SET_NAME wasn't specified"
                )
            endif()
            set(export_args)
        endif()
        install(
            TARGETS ${target} ${export_args}
            ARCHIVE DESTINATION ${archive_subdir}
            ${ARGN}
            LIBRARY DESTINATION ${library_subdir}
            ${ARGN}
            RUNTIME DESTINATION ${runtime_subdir}
            ${ARGN}
            PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
            ${ARGN}
        )
    endmacro()

    if(ARG_INSTALL_COMPONENT)
        i(EXCLUDE_FROM_ALL COMPONENT ${ARG_INSTALL_COMPONENT})
        set(debug_component "${ARG_INSTALL_COMPONENT}-debug-info")
    elseif(ARG_INSTALL)
        i()
        set(debug_component "debug-info")
    endif()

    if(DEFINED ARG_DEBUG_INFO_INSTALL_COMPONENT)
        set(debug_component "${ARG_DEBUG_INFO_INSTALL_COMPONENT}")
    endif()

    # Install debug info only if target is being installed
    if((ARG_INSTALL OR ARG_INSTALL_COMPONENT) AND generate_split_debug_info)
        if(type STREQUAL "EXECUTABLE" OR WIN32)
            set(debug_dest ${runtime_subdir})
        else()
            set(debug_dest ${library_subdir})
        endif()

        if(MSVC)
            set(debug_file $<TARGET_PDB_FILE:${target}>)
        else()
            set(debug_file "$<TARGET_FILE:${target}>.dwarf")
        endif()

        install(
            FILES ${debug_file}
            DESTINATION ${debug_dest}
            COMPONENT ${debug_component}
            EXCLUDE_FROM_ALL
            OPTIONAL
        )
    endif()
endfunction()

# Ideally we'd use CMAKE_INSTALL_LIBDIR and CMAKE_INSTALL_RUNTIMEDIR here,
# however some Slang functionality (specifically generating executables on
# Linux systems) relies on runtime libraries being at "$ORIGIN/../lib". This
# could be improved by setting at configure-time that path to be the relative
# path from CM_I_RD to CM_I_LD.
set(library_subdir lib)
set(runtime_subdir bin)

# On Windows, because there's no RPATH, place modules in bin, next to the
# executables which load them (by deault, CMAKE will place them in lib and
# expect the application to seek them out there)
#
# This variable is used in the above function as and elsewhere for installing
# an imported module (slang-llvm from binary), hence why it's defined here.
if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    set(module_subdir ${runtime_subdir})
else()
    set(module_subdir ${library_subdir})
endif()
