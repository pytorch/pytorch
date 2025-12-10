# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

# Author: Alex Turbov

if(CMAKE_BINARY_DIR)
  message(FATAL_ERROR "CPackNuGet.cmake may only be used by CPack internally.")
endif()

function(_cpack_nuget_debug)
    if(CPACK_NUGET_PACKAGE_DEBUG)
        message("CPackNuGet:Debug: " ${ARGN})
    endif()
endfunction()

function(_cpack_nuget_debug_var NAME)
    if(CPACK_NUGET_PACKAGE_DEBUG)
        message("CPackNuGet:Debug: ${NAME}=`${${NAME}}`")
    endif()
endfunction()

function(_cpack_nuget_variable_fallback OUTPUT_VAR_NAME NUGET_VAR_NAME)
    if(ARGN)
        list(JOIN ARGN "`, `" _va_args)
        set(_va_args ", ARGN: `${_va_args}`")
    endif()
    _cpack_nuget_debug(
        "_cpack_nuget_variable_fallback: "
        "OUTPUT_VAR_NAME=`${OUTPUT_VAR_NAME}`, "
        "NUGET_VAR_NAME=`${NUGET_VAR_NAME}`"
        "${_va_args}"
      )

    set(_options USE_CDATA)
    set(_one_value_args LIST_GLUE)
    set(_multi_value_args FALLBACK_VARS)
    cmake_parse_arguments(PARSE_ARGV 0 _args "${_options}" "${_one_value_args}" "${_multi_value_args}")

    if(CPACK_NUGET_PACKAGE_COMPONENT)
        string(
            TOUPPER "${CPACK_NUGET_PACKAGE_COMPONENT}"
            CPACK_NUGET_PACKAGE_COMPONENT_UPPER
          )
    endif()

    if(CPACK_NUGET_PACKAGE_COMPONENT
      AND CPACK_NUGET_${CPACK_NUGET_PACKAGE_COMPONENT}_PACKAGE_${NUGET_VAR_NAME}
      )
        set(
            _result
            "${CPACK_NUGET_${CPACK_NUGET_PACKAGE_COMPONENT}_PACKAGE_${NUGET_VAR_NAME}}"
          )
        _cpack_nuget_debug(
            "  CPACK_NUGET_${CPACK_NUGET_PACKAGE_COMPONENT}_PACKAGE_${NUGET_VAR_NAME}: "
            "OUTPUT_VAR_NAME->${OUTPUT_VAR_NAME}=`${_result}`"
          )

    elseif(CPACK_NUGET_PACKAGE_COMPONENT_UPPER
      AND CPACK_NUGET_${CPACK_NUGET_PACKAGE_COMPONENT_UPPER}_PACKAGE_${NUGET_VAR_NAME}
      )
        set(
            _result
            "${CPACK_NUGET_${CPACK_NUGET_PACKAGE_COMPONENT_UPPER}_PACKAGE_${NUGET_VAR_NAME}}"
          )
        _cpack_nuget_debug(
            "  CPACK_NUGET_${CPACK_NUGET_PACKAGE_COMPONENT_UPPER}_PACKAGE_${NUGET_VAR_NAME}: "
            "OUTPUT_VAR_NAME->${OUTPUT_VAR_NAME}=`${_result}`"
          )

    elseif(CPACK_NUGET_PACKAGE_${NUGET_VAR_NAME})
        set(_result "${CPACK_NUGET_PACKAGE_${NUGET_VAR_NAME}}")
        _cpack_nuget_debug(
            "  CPACK_NUGET_PACKAGE_${NUGET_VAR_NAME}: "
            "OUTPUT_VAR_NAME->${OUTPUT_VAR_NAME}=`${_result}`"
          )

    else()
        foreach(_var IN LISTS _args_FALLBACK_VARS)
            _cpack_nuget_debug("  Fallback: ${_var} ...")
            if(${_var})
                _cpack_nuget_debug("            ${_var}=`${${_var}}`")
                set(_result "${${_var}}")
                _cpack_nuget_debug(
                    "  ${_var}: OUTPUT_VAR_NAME->${OUTPUT_VAR_NAME}=`${_result}`"
                  )
                break()
            endif()
        endforeach()
    endif()

    if(_result)
        if(_args_USE_CDATA)
            set(_value_before "<![CDATA[")
            set(_value_after "]]>")
        endif()

        list(LENGTH _result _result_len)
        if(_result_len GREATER 1 AND _args_LIST_GLUE)
            list(JOIN _result "${_args_LIST_GLUE}" _result)
        endif()

        set(${OUTPUT_VAR_NAME} "${_value_before}${_result}${_value_after}" PARENT_SCOPE)
    endif()

endfunction()

function(_cpack_nuget_variable_fallback_and_wrap_into_element ELEMENT NUGET_VAR_NAME)
    set(_options)
    set(_one_value_args)
    set(_multi_value_args FALLBACK_VARS ATTRIBUTES)
    cmake_parse_arguments(PARSE_ARGV 2 _args "${_options}" "${_one_value_args}" "${_multi_value_args}")

    if(_args_ATTRIBUTES)
        list(JOIN _args_ATTRIBUTES " " _attributes)
        string(PREPEND _attributes " ")
    endif()

    _cpack_nuget_variable_fallback(_value ${NUGET_VAR_NAME} ${ARGN} USE_CDATA)

    string(TOUPPER "${ELEMENT}" _ELEMENT_UP)
    if(_value)
        set(
            _CPACK_NUGET_${_ELEMENT_UP}_TAG
            "<${ELEMENT}${_attributes}>${_value}</${ELEMENT}>"
            PARENT_SCOPE
          )
    elseif(_attributes)
        set(
            _CPACK_NUGET_${_ELEMENT_UP}_TAG
            "<${ELEMENT}${_attributes} />"
            PARENT_SCOPE
          )
    endif()
endfunction()

# Warn of obsolete nuspec fields, referencing CMake variables and suggested
# replacement, if any
function(_cpack_nuget_deprecation_warning NUGET_ELEMENT VARNAME REPLACEMENT)
    if(${VARNAME})
        if(REPLACEMENT)
            message(DEPRECATION "nuspec element `${NUGET_ELEMENT}` is deprecated in NuGet; consider replacing `${VARNAME}` with `${REPLACEMENT}`")
        else()
            message(DEPRECATION "nuspec element `${NUGET_ELEMENT}` is deprecated in NuGet; consider removing `${VARNAME}`")
        endif()
    endif()
endfunction()

# Print some debug info
_cpack_nuget_debug("---[CPack NuGet Input Variables]---")
_cpack_nuget_debug_var(CPACK_PACKAGE_NAME)
_cpack_nuget_debug_var(CPACK_PACKAGE_VERSION)
_cpack_nuget_debug_var(CPACK_TOPLEVEL_TAG)
_cpack_nuget_debug_var(CPACK_TOPLEVEL_DIRECTORY)
_cpack_nuget_debug_var(CPACK_TEMPORARY_DIRECTORY)
_cpack_nuget_debug_var(CPACK_NUGET_GROUPS)
if(CPACK_NUGET_GROUPS)
    foreach(_group IN LISTS CPACK_NUGET_GROUPS)
        string(MAKE_C_IDENTIFIER "${_group}" _group_up)
        string(TOUPPER "${_group_up}" _group_up)
        _cpack_nuget_debug_var(CPACK_NUGET_${_group_up}_GROUP_COMPONENTS)
    endforeach()
endif()
_cpack_nuget_debug_var(CPACK_NUGET_COMPONENTS)
_cpack_nuget_debug_var(CPACK_NUGET_ALL_IN_ONE)
_cpack_nuget_debug_var(CPACK_NUGET_ORDINAL_MONOLITIC)
_cpack_nuget_debug("-----------------------------------")

function(_cpack_nuget_render_spec)
    # Make a variable w/ upper-cased component name
    if(CPACK_NUGET_PACKAGE_COMPONENT)
        string(TOUPPER "${CPACK_NUGET_PACKAGE_COMPONENT}" CPACK_NUGET_PACKAGE_COMPONENT_UPPER)
    endif()

    # Set mandatory variables (not wrapped into XML elements)
    # https://docs.microsoft.com/en-us/nuget/reference/nuspec#required-metadata-elements
    if(CPACK_NUGET_PACKAGE_COMPONENT)
        if(CPACK_NUGET_${CPACK_NUGET_PACKAGE_COMPONENT_UPPER}_PACKAGE_NAME)
            set(
                CPACK_NUGET_PACKAGE_NAME
                "${CPACK_NUGET_${CPACK_NUGET_PACKAGE_COMPONENT_UPPER}_PACKAGE_NAME}"
              )
        elseif(NOT CPACK_NUGET_PACKAGE_COMPONENT STREQUAL "Unspecified")
            set(
                CPACK_NUGET_PACKAGE_NAME
                "${CPACK_PACKAGE_NAME}.${CPACK_NUGET_PACKAGE_COMPONENT}"
              )
        else()
            set(CPACK_NUGET_PACKAGE_NAME "${CPACK_PACKAGE_NAME}")
        endif()
    elseif(NOT CPACK_NUGET_PACKAGE_NAME)
        set(CPACK_NUGET_PACKAGE_NAME "${CPACK_PACKAGE_NAME}")
    endif()

    # Warn about deprecated nuspec elements; warnings only display if
    # variable is set
    # Note that while nuspec's "summary" element is deprecated, there
    # is no suggested replacement so (for now) no deprecation warning
    # is shown for `CPACK_NUGET_*_DESCRIPTION_SUMMARY`
    _cpack_nuget_deprecation_warning("licenseUrl" CPACK_NUGET_PACKAGE_LICENSEURL
        "CPACK_NUGET_PACKAGE_LICENSE_FILE_NAME or CPACK_NUGET_PACKAGE_LICENSE_EXPRESSION")
    _cpack_nuget_deprecation_warning("licenseUrl" CPACK_NUGET_${CPACK_NUGET_PACKAGE_COMPONENT}_LICENSEURL
        "CPACK_NUGET_${CPACK_NUGET_PACKAGE_COMPONENT}_LICENSE_FILE_NAME or CPACK_NUGET_${CPACK_NUGET_PACKAGE_COMPONENT}_LICENSE_EXPRESSION")
    _cpack_nuget_deprecation_warning("iconUrl" CPACK_NUGET_PACKAGE_ICONURL
        "CPACK_NUGET_PACKAGE_ICON")
    _cpack_nuget_deprecation_warning("iconUrl" CPACK_NUGET_${CPACK_NUGET_PACKAGE_COMPONENT}_ICONURL
        "CPACK_NUGET_${CPACK_NUGET_PACKAGE_COMPONENT}_ICON")

    # Set nuspec fields
    _cpack_nuget_variable_fallback(
        CPACK_NUGET_PACKAGE_VERSION VERSION
        FALLBACK_VARS
            CPACK_PACKAGE_VERSION
      )
    _cpack_nuget_variable_fallback(
        CPACK_NUGET_PACKAGE_DESCRIPTION DESCRIPTION
        FALLBACK_VARS
            CPACK_COMPONENT_${CPACK_NUGET_PACKAGE_COMPONENT}_DESCRIPTION
            CPACK_COMPONENT_${CPACK_NUGET_PACKAGE_COMPONENT_UPPER}_DESCRIPTION
            CPACK_COMPONENT_GROUP_${CPACK_NUGET_PACKAGE_COMPONENT_UPPER}_DESCRIPTION
            CPACK_PACKAGE_DESCRIPTION
        USE_CDATA
      )
    _cpack_nuget_variable_fallback(
        CPACK_NUGET_PACKAGE_AUTHORS AUTHORS
        FALLBACK_VARS
            CPACK_PACKAGE_VENDOR
        USE_CDATA
        LIST_GLUE ","
      )

    # Set optional variables (wrapped into XML elements)
    # https://docs.microsoft.com/en-us/nuget/reference/nuspec#optional-metadata-elements
    _cpack_nuget_variable_fallback_and_wrap_into_element(
        title
        TITLE
        FALLBACK_VARS
            CPACK_COMPONENT_${CPACK_NUGET_PACKAGE_COMPONENT}_DISPLAY_NAME
            CPACK_COMPONENT_${CPACK_NUGET_PACKAGE_COMPONENT_UPPER}_DISPLAY_NAME
            CPACK_COMPONENT_GROUP_${CPACK_NUGET_PACKAGE_COMPONENT_UPPER}_DISPLAY_NAME
      )
    _cpack_nuget_variable_fallback_and_wrap_into_element(owners OWNERS LIST_GLUE ",")
    _cpack_nuget_variable_fallback_and_wrap_into_element(
        projectUrl
        HOMEPAGE_URL
        FALLBACK_VARS
            CPACK_PACKAGE_HOMEPAGE_URL
      )

    # "licenseUrl" is deprecated in favor of "license"
    _cpack_nuget_variable_fallback_and_wrap_into_element(licenseUrl LICENSEURL)

    # "iconUrl" is deprecated in favor of "icon"
    _cpack_nuget_variable_fallback_and_wrap_into_element(iconUrl ICONURL)

    # "license" takes a "type" attribute of either "file" or "expression"
    # "file" refers to a file path of a .txt or .md file relative to the installation root
    # "expression" refers to simple or compound expression of license identifiers
    # listed at https://spdx.org/licenses/
    # Note that only one of CPACK_NUGET_PACKAGE_LICENSE_FILE_NAME and
    # CPACK_NUGET_PACKAGE_LICENSE_EXPRESSION may be specified. If both are specified,
    # CPACK_NUGET_PACKAGE_LICENSE_FILE_NAME takes precedence and CPACK_NUGET_PACKAGE_LICENSE_EXPRESSION is ignored.
    if(CPACK_NUGET_PACKAGE_LICENSE_FILE_NAME)
        _cpack_nuget_variable_fallback_and_wrap_into_element(
            license LICENSE_FILE_NAME
            ATTRIBUTES [[type="file"]]
          )
    elseif(CPACK_NUGET_PACKAGE_LICENSE_EXPRESSION)
        _cpack_nuget_variable_fallback_and_wrap_into_element(
            license LICENSE_EXPRESSION
            ATTRIBUTES [[type="expression"]]
          )
    endif()

    # "icon" refers to a file path relative to the installation root
    _cpack_nuget_variable_fallback_and_wrap_into_element(icon ICON)
    # "summary" is deprecated in favor of "description"
    _cpack_nuget_variable_fallback_and_wrap_into_element(
        summary DESCRIPTION_SUMMARY
        FALLBACK_VARS
            CPACK_PACKAGE_DESCRIPTION_SUMMARY
      )
    if(CPACK_NUGET_PACKAGE_REQUIRE_LICENSE_ACCEPTANCE)
        set(
            _CPACK_NUGET_REQUIRELICENSEACCEPTANCE_TAG
            "<requireLicenseAcceptance>true</requireLicenseAcceptance>"
          )
    endif()
    _cpack_nuget_variable_fallback_and_wrap_into_element(releaseNotes RELEASE_NOTES)
    _cpack_nuget_variable_fallback_and_wrap_into_element(copyright COPYRIGHT)
    # "language" is a locale identifier such as "en_CA"
    _cpack_nuget_variable_fallback_and_wrap_into_element(language LANGUAGE)
    _cpack_nuget_variable_fallback_and_wrap_into_element(tags TAGS LIST_GLUE " ")
    # "repository" holds repository metadata consisting of four optional
    # attributes: "type", "url", "branch", and "commit". While all fields are
    # considered optional, they are not independent. Currently unsupported.

    # NuGet >= 5.10
    _cpack_nuget_variable_fallback_and_wrap_into_element(readme README)

    set(_CPACK_NUGET_REPOSITORY_TAG)
    _cpack_nuget_variable_fallback(_repo_type REPOSITORY_TYPE)
    _cpack_nuget_variable_fallback(_repo_url REPOSITORY_URL)
    if(_repo_type AND _repo_url)
        set(_CPACK_NUGET_REPOSITORY_TAG "<repository type=\"${_repo_type}\" url=\"${_repo_url}\"")
        _cpack_nuget_variable_fallback(_repo_br REPOSITORY_BRANCH)
        if(_repo_br)
            string(APPEND _CPACK_NUGET_REPOSITORY_TAG " branch=\"${_repo_br}\"")
        endif()
        _cpack_nuget_variable_fallback(_repo_commit REPOSITORY_COMMIT)
        if(_repo_commit)
            string(APPEND _CPACK_NUGET_REPOSITORY_TAG " commit=\"${_repo_commit}\"")
        endif()
        string(APPEND _CPACK_NUGET_REPOSITORY_TAG " />")
    else()
        message(AUTHOR_WARNING "Skip adding the `<repository .../>` element due to missing URL or type")
    endif()

    # Handle dependencies
    # Primary deps (not specific to any framework)
    _cpack_nuget_render_deps_group("" rendered_group)
    string(APPEND _CPACK_NUGET_DEPENDENCIES_TAG "${rendered_group}")

    # Framework-specific deps
    _cpack_nuget_variable_fallback(_tfms TFMS)
    foreach(tfm IN LISTS _tfms)
        _cpack_nuget_render_deps_group("${tfm}" rendered_group)
        string(APPEND _CPACK_NUGET_DEPENDENCIES_TAG "${rendered_group}")
    endforeach()

    # If there are any dependencies to include, wrap them with the appropriate tag
    if(_CPACK_NUGET_DEPENDENCIES_TAG)
        string(PREPEND _CPACK_NUGET_DEPENDENCIES_TAG "<dependencies>\n")
        string(APPEND _CPACK_NUGET_DEPENDENCIES_TAG "        </dependencies>")
    endif()

    # Render the spec file
    # NOTE The spec filename doesn't matter. Being included into a package,
    # NuGet will name it properly.
    _cpack_nuget_debug("Rendering `${CPACK_TEMPORARY_DIRECTORY}/CPack.NuGet.nuspec` file...")
    configure_file(
        "${CMAKE_ROOT}/Modules/Internal/CPack/CPack.NuGet.nuspec.in"
        "${CPACK_TEMPORARY_DIRECTORY}/CPack.NuGet.nuspec"
        @ONLY
      )
endfunction()

# Call this function once for each TWFM (e.g., 'net48') to generate the dependencies for
# that framework. It can also be called with an empty TWFM for the "general" set of
# dependencies, which are not specific to a framework.
function(_cpack_nuget_render_deps_group TFM OUTPUT_VAR_NAME)
    _cpack_nuget_debug("  rendering deps for ${TFM}")
    if(TFM)
        set(_tfm "_${TFM}")
    else()
        set(_tfm "")
    endif()
    _cpack_nuget_variable_fallback(_deps DEPENDENCIES${_tfm})
    set(_collected_deps)
    foreach(_dep IN LISTS _deps)
        set(_ver)  # Ensure we don't accidentally use the version from the previous dep in the list
        _cpack_nuget_debug("  checking dependency `${_dep}`")

        _cpack_nuget_variable_fallback(_ver DEPENDENCIES${_tfm}_${_dep}_VERSION)

        if(NOT _ver)
            string(TOUPPER "${_dep}" _dep_upper)
            _cpack_nuget_variable_fallback(_ver DEPENDENCIES${_tfm}_${_dep_upper}_VERSION)
        endif()

        if(_ver)
            _cpack_nuget_debug("  got `${_dep}` dependency version ${_ver}")
            string(APPEND _collected_deps "                <dependency id=\"${_dep}\" version=\"${_ver}\" />\n")
        endif()
    endforeach()

    # Render deps into the variable
    if(TFM)
        _cpack_nuget_convert_tfm_to_frameworkname("${TFM}" framework_name)
        _cpack_nuget_debug("  converted ${TFM} to ${framework_name}")
    endif()
    set(rendered_group)
    if(_collected_deps)
        if(TFM)
            _cpack_nuget_debug("  rendering group for framework ${framework_name}")
            string(CONCAT rendered_group "            <group targetFramework=\"${framework_name}\">\n" "${_collected_deps}" "            </group>\n")
        else()
            _cpack_nuget_debug("  rendering primary group")
            string(CONCAT rendered_group "            <group>\n" "${_collected_deps}" "            </group>\n")
        endif()
    elseif(TFM)
        _cpack_nuget_debug("  no deps for ${TFM}, rendering empty group")
        # Insert an empty group for a framework that doesn't have any specific dependencies listed, as the existence
        # of this group can be used by NuGet to see that the framework is supported.
        string(CONCAT rendered_group "            <group targetFramework=\"${framework_name}\" />\n")
    endif()
    set(${OUTPUT_VAR_NAME} "${rendered_group}" PARENT_SCOPE)
endfunction()

# Tries to look up a Framework Name (e.g., '.NETFramework4.8') from a Target Framework Moniker (TFM) (e.g., 'net48')
function(_cpack_nuget_convert_tfm_to_frameworkname TFM OUTPUT_VAR_NAME)
    # There are a few patterns to handle:
    # 1a. net4          -> .NETFramework4
    # 1b. net5.0        -> net5.0            From version 5 onwards, the name just looks the same as the moniker, and both need to have a dot
    # 2. netstandard13  -> .NETStandard1.3
    # 3. netcoreapp21   -> .NETCoreApp2.1
    # 4. dotnet50       -> .NETPlatform5.0
    # 5. native0.0      -> native0.0         Support for native C++ and mixed C++/CLI projects
    if(TFM MATCHES "^net([1-4](.[\.0-9])?)$") # CMAKE_MATCH_1 holds the version part
        _cpack_nuget_get_dotted_version("${CMAKE_MATCH_1}" dotted_version)
        set(framework_name ".NETFramework${dotted_version}")
    elseif(TFM MATCHES "^net[1-9](\.[0-9]+)+$")
        set(framework_name "${TFM}")
    elseif(TFM MATCHES "^netstandard([0-9]+(\.[0-9]+)*)$")
        _cpack_nuget_get_dotted_version("${CMAKE_MATCH_1}" dotted_version)
        set(framework_name ".NETStandard${dotted_version}")
    elseif(TFM MATCHES "^netcoreapp([0-9]+(\.[0-9]+)*)$")
        _cpack_nuget_get_dotted_version("${CMAKE_MATCH_1}" dotted_version)
        set(framework_name ".NETCoreApp${dotted_version}")
    elseif(TFM MATCHES "^dotnet([0-9]+(\.[0-9]+)*)$")
        _cpack_nuget_get_dotted_version("${CMAKE_MATCH_1}" dotted_version)
        set(framework_name ".NETPlatform${dotted_version}")
    elseif(TFM STREQUAL "native0.0")
        set(framework_name "${TFM}")
    else()
        message(FATAL_ERROR "Target Framework Moniker '${TFM}' not recognized")
    endif()
    set(${OUTPUT_VAR_NAME} ${framework_name} PARENT_SCOPE)
endfunction()

function(_cpack_nuget_get_dotted_version VERSION OUTPUT_VAR_NAME)
    if(VERSION MATCHES "\.")
        # The version already has dots in it, just reuse the numbers given
        set(dotted_version "${VERSION}")
    else()
        # No dots in the version, treat each digit as a version part
        string(LENGTH "${VERSION}" length)
        math(EXPR last_index "${length} - 1")
        string(SUBSTRING "${VERSION}" 0 1 digit)
        set(dotted_version "${digit}")
        foreach(i RANGE 1 ${last_index})
            string(SUBSTRING "${VERSION}" ${i} 1 digit)
            string(APPEND dotted_version ".${digit}")
        endforeach()
    endif()
    # This would be a good place to remove any superfluous ".0"s from the end of the version string, but
    # for now it should be fine to just expect the caller not to supply them in the first place.
    set(${OUTPUT_VAR_NAME} "${dotted_version}" PARENT_SCOPE)
endfunction()

function(_cpack_nuget_make_files_tag)
    set(_files)
    foreach(_comp IN LISTS ARGN)
        cmake_path(APPEND _comp "**")
        cmake_path(NATIVE_PATH _comp _comp)
        string(APPEND _files "        <file src=\"${_comp}\" target=\".\" />\n")
    endforeach()
    set(_CPACK_NUGET_FILES_TAG "<files>\n${_files}    </files>" PARENT_SCOPE)
endfunction()

find_program(NUGET_EXECUTABLE nuget)
_cpack_nuget_debug_var(NUGET_EXECUTABLE)
if(NOT NUGET_EXECUTABLE)
    message(FATAL_ERROR "NuGet executable not found")
endif()

# Add details for debug run
if(CPACK_NUGET_PACKAGE_DEBUG)
    list(APPEND CPACK_NUGET_PACK_ADDITIONAL_OPTIONS "-Verbosity" "detailed")
endif()

# Generate symbol package
if(CPACK_NUGET_SYMBOL_PACKAGE)
    list(APPEND CPACK_NUGET_PACK_ADDITIONAL_OPTIONS "-Symbols")
    list(APPEND CPACK_NUGET_PACK_ADDITIONAL_OPTIONS "-SymbolPackageFormat" "snupkg")
endif()

# Case one: ordinal all-in-one package
if(CPACK_NUGET_ORDINAL_MONOLITIC)
    # This variable `CPACK_NUGET_ALL_IN_ONE` set by C++ code:
    # Meaning to pack all installed files into a single package
    _cpack_nuget_debug("---[Making an ordinal monolitic package]---")
    _cpack_nuget_render_spec()
    execute_process(
        COMMAND "${NUGET_EXECUTABLE}" pack ${CPACK_NUGET_PACK_ADDITIONAL_OPTIONS}
        WORKING_DIRECTORY "${CPACK_TEMPORARY_DIRECTORY}"
        RESULT_VARIABLE _nuget_result
      )
    if(NOT _nuget_result EQUAL 0)
        message(FATAL_ERROR "Nuget pack failed")
    endif()

elseif(CPACK_NUGET_ALL_IN_ONE)
    # This variable `CPACK_NUGET_ALL_IN_ONE` set by C++ code:
    # Meaning to pack all installed components into a single package
    _cpack_nuget_debug("---[Making a monolitic package from installed components]---")

    # Prepare the `files` element which include files from several components
    _cpack_nuget_make_files_tag(${CPACK_NUGET_COMPONENTS})
    _cpack_nuget_render_spec()
    execute_process(
        COMMAND "${NUGET_EXECUTABLE}" pack ${CPACK_NUGET_PACK_ADDITIONAL_OPTIONS}
        WORKING_DIRECTORY "${CPACK_TEMPORARY_DIRECTORY}"
        RESULT_VARIABLE _nuget_result
      )
    if(NOT _nuget_result EQUAL 0)
        message(FATAL_ERROR "Nuget pack failed")
    endif()

else()
    # Is there any grouped component?
    if(CPACK_NUGET_GROUPS)
        _cpack_nuget_debug("---[Making grouped component(s) package(s)]---")
        foreach(_group IN LISTS CPACK_NUGET_GROUPS)
            _cpack_nuget_debug("Starting to make the package for group `${_group}`")
            string(MAKE_C_IDENTIFIER "${_group}" _group_up)
            string(TOUPPER "${_group_up}" _group_up)

            # Render a spec file which includes all components in the current group
            unset(_CPACK_NUGET_FILES_TAG)
            _cpack_nuget_make_files_tag(${CPACK_NUGET_${_group_up}_GROUP_COMPONENTS})
            # Temporary set `CPACK_NUGET_PACKAGE_COMPONENT` to the group name
            # to properly collect various per group settings
            set(CPACK_NUGET_PACKAGE_COMPONENT ${_group})
            _cpack_nuget_render_spec()
            unset(CPACK_NUGET_PACKAGE_COMPONENT)
            execute_process(
                COMMAND "${NUGET_EXECUTABLE}" pack ${CPACK_NUGET_PACK_ADDITIONAL_OPTIONS}
                WORKING_DIRECTORY "${CPACK_TEMPORARY_DIRECTORY}"
                RESULT_VARIABLE _nuget_result
              )
            if(NOT _nuget_result EQUAL 0)
                message(FATAL_ERROR "Nuget pack failed")
            endif()
        endforeach()
    endif()
    # Is there any single component package needed?
    if(CPACK_NUGET_COMPONENTS)
        _cpack_nuget_debug("---[Making single-component(s) package(s)]---")
        foreach(_comp IN LISTS CPACK_NUGET_COMPONENTS)
            _cpack_nuget_debug("Starting to make the package for component `${_comp}`")
            # Render a spec file which includes only given component
            unset(_CPACK_NUGET_FILES_TAG)
            _cpack_nuget_make_files_tag(${_comp})
            # Temporary set `CPACK_NUGET_PACKAGE_COMPONENT` to the current
            # component name to properly collect various per group settings
            set(CPACK_NUGET_PACKAGE_COMPONENT ${_comp})
            _cpack_nuget_render_spec()
            unset(CPACK_NUGET_PACKAGE_COMPONENT)
            execute_process(
                COMMAND "${NUGET_EXECUTABLE}" pack ${CPACK_NUGET_PACK_ADDITIONAL_OPTIONS}
                WORKING_DIRECTORY "${CPACK_TEMPORARY_DIRECTORY}"
                RESULT_VARIABLE _nuget_result
              )
            if(NOT _nuget_result EQUAL 0)
                message(FATAL_ERROR "Nuget pack failed")
            endif()
        endforeach()
    endif()
endif()


file(GLOB_RECURSE GEN_CPACK_NUGET_PACKAGE_FILES "${CPACK_TEMPORARY_DIRECTORY}/*.nupkg")
if(NOT GEN_CPACK_NUGET_PACKAGE_FILES)
    message(FATAL_ERROR "NuGet package was not generated at `${CPACK_TEMPORARY_DIRECTORY}`!")
endif()
list(APPEND GEN_CPACK_OUTPUT_FILES "${GEN_CPACK_NUGET_PACKAGE_FILES}")

if(CPACK_NUGET_SYMBOL_PACKAGE)
  file(GLOB_RECURSE GEN_CPACK_NUGET_SYMBOL_PACKAGE_FILES "${CPACK_TEMPORARY_DIRECTORY}/*.snupkg")
  if(NOT GEN_CPACK_NUGET_SYMBOL_PACKAGE_FILES)
        message(FATAL_ERROR "NuGet symbol package was not generated at `${CPACK_TEMPORARY_DIRECTORY}`!")
    endif()
    list(APPEND GEN_CPACK_OUTPUT_FILES "${GEN_CPACK_NUGET_SYMBOL_PACKAGE_FILES}")
endif()

_cpack_nuget_debug("Generated files: ${GEN_CPACK_OUTPUT_FILES}")
