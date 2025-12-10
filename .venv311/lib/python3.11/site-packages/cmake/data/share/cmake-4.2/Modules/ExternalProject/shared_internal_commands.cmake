cmake_policy(VERSION 4.1)

# Determine the remote URL of the project containing the working_directory.
# This will leave output_variable unset if the URL can't be determined.
function(_ep_get_git_remote_url output_variable working_directory)
  set("${output_variable}" "" PARENT_SCOPE)

  find_package(Git QUIET REQUIRED)

  execute_process(
    COMMAND ${GIT_EXECUTABLE} symbolic-ref --short HEAD
    WORKING_DIRECTORY "${working_directory}"
    OUTPUT_VARIABLE git_symbolic_ref
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
  )

  if(NOT git_symbolic_ref STREQUAL "")
    # We are potentially on a branch. See if that branch is associated with
    # an upstream remote (might be just a local one or not a branch at all).
    execute_process(
      COMMAND ${GIT_EXECUTABLE} config branch.${git_symbolic_ref}.remote
      WORKING_DIRECTORY "${working_directory}"
      OUTPUT_VARIABLE git_remote_name
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET
    )
  endif()

  if(NOT git_remote_name)
    # Can't select a remote based on a branch. If there's only one remote,
    # or we have multiple remotes but one is called "origin", choose that.
    execute_process(
      COMMAND ${GIT_EXECUTABLE} remote
      WORKING_DIRECTORY "${working_directory}"
      OUTPUT_VARIABLE git_remote_list
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET
    )
    string(REPLACE "\n" ";" git_remote_list "${git_remote_list}")
    list(LENGTH git_remote_list git_remote_list_length)

    if(git_remote_list_length EQUAL 0)
      message(FATAL_ERROR "Git remote not found in parent project.")
    elseif(git_remote_list_length EQUAL 1)
      list(GET git_remote_list 0 git_remote_name)
    else()
      set(base_warning_msg "Multiple git remotes found for parent project")
      if("origin" IN_LIST git_remote_list)
        message(WARNING "${base_warning_msg}, defaulting to origin.")
        set(git_remote_name "origin")
      else()
        message(FATAL_ERROR "${base_warning_msg}, none of which are origin.")
      endif()
    endif()
  endif()

  if(Git_VERSION VERSION_LESS 1.7.5)
    set(_git_remote_url_cmd_args config remote.${git_remote_name}.url)
  elseif(Git_VERSION VERSION_LESS 2.7)
    set(_git_remote_url_cmd_args ls-remote --get-url ${git_remote_name})
  else()
    set(_git_remote_url_cmd_args remote get-url ${git_remote_name})
  endif()

  execute_process(
    COMMAND ${GIT_EXECUTABLE} ${_git_remote_url_cmd_args}
    WORKING_DIRECTORY "${working_directory}"
    OUTPUT_VARIABLE git_remote_url
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL LAST
    ENCODING UTF-8   # Needed to handle non-ascii characters in local paths
  )

  set("${output_variable}" "${git_remote_url}" PARENT_SCOPE)
endfunction()


function(_ep_is_relative_git_remote output_variable remote_url)
  if(remote_url MATCHES "^\\.\\./")
    set("${output_variable}" TRUE PARENT_SCOPE)
  else()
    set("${output_variable}" FALSE PARENT_SCOPE)
  endif()
endfunction()


# Return an absolute remote URL given an existing remote URL and relative path.
# The output_variable will be set to an empty string if an absolute URL
# could not be computed (no error message is output).
function(_ep_resolve_relative_git_remote
  output_variable
  parent_remote_url
  relative_remote_url
)
  set("${output_variable}" "" PARENT_SCOPE)

  if(parent_remote_url STREQUAL "")
    return()
  endif()

  string(REGEX MATCH
    "^(([A-Za-z0-9][A-Za-z0-9+.-]*)://)?(([^/@]+)@)?(\\[[A-Za-z0-9:]+\\]|[^/:]+)?([/:]/?)(.+(\\.git)?/?)$"
    git_remote_url_components
    "${parent_remote_url}"
  )

  set(protocol "${CMAKE_MATCH_1}")
  set(auth "${CMAKE_MATCH_3}")
  set(host "${CMAKE_MATCH_5}")
  set(separator "${CMAKE_MATCH_6}")
  set(path "${CMAKE_MATCH_7}")

  string(REPLACE "/" ";" remote_path_components "${path}")
  string(REPLACE "/" ";" relative_path_components "${relative_remote_url}")

  foreach(relative_path_component IN LISTS relative_path_components)
    if(NOT relative_path_component STREQUAL "..")
      break()
    endif()

    list(LENGTH remote_path_components remote_path_component_count)

    if(remote_path_component_count LESS 1)
      return()
    endif()

    list(POP_BACK remote_path_components)
    list(POP_FRONT relative_path_components)
  endforeach()

  list(APPEND final_path_components ${remote_path_components} ${relative_path_components})
  list(JOIN final_path_components "/" path)

  set("${output_variable}" "${protocol}${auth}${host}${separator}${path}" PARENT_SCOPE)
endfunction()


# The output_variable will be set to the original git_repository if it
# could not be resolved (no error message is output). The original value is
# also returned if it doesn't need to be resolved.
function(_ep_resolve_git_remote
  output_variable
  git_repository
  cmp0150
  cmp0150_old_base_dir
)
  if(git_repository STREQUAL "")
    set("${output_variable}" "" PARENT_SCOPE)
    return()
  endif()

  _ep_is_relative_git_remote(_git_repository_is_relative "${git_repository}")

  if(NOT _git_repository_is_relative)
    set("${output_variable}" "${git_repository}" PARENT_SCOPE)
    return()
  endif()

  if(cmp0150 STREQUAL "NEW")
    _ep_get_git_remote_url(_parent_git_remote_url "${CMAKE_CURRENT_SOURCE_DIR}")
    _ep_resolve_relative_git_remote(_resolved_git_remote_url "${_parent_git_remote_url}" "${git_repository}")

    if(_resolved_git_remote_url STREQUAL "")
      message(FATAL_ERROR
        "Failed to resolve relative git remote URL:\n"
        "  Relative URL: ${git_repository}\n"
        "  Parent URL:   ${_parent_git_remote_url}"
      )
    endif()
    set("${output_variable}" "${_resolved_git_remote_url}" PARENT_SCOPE)
    return()
  elseif(cmp0150 STREQUAL "")
    cmake_policy(GET_WARNING CMP0150 _cmp0150_warning)
    message(AUTHOR_WARNING
      "${_cmp0150_warning}\n"
      "A relative GIT_REPOSITORY path was detected. "
      "This will be interpreted as a local path to where the project is being cloned. "
      "Set GIT_REPOSITORY to an absolute path or set policy CMP0150 to NEW to avoid "
      "this warning."
    )
  endif()

  set("${output_variable}" "${cmp0150_old_base_dir}/${git_repository}" PARENT_SCOPE)
endfunction()


macro(_ep_get_hash_algos out_var)
  set(${out_var}
    MD5
    SHA1
    SHA224
    SHA256
    SHA384
    SHA512
    SHA3_224
    SHA3_256
    SHA3_384
    SHA3_512
  )
endmacro()


macro(_ep_get_hash_regex out_var)
  _ep_get_hash_algos(${out_var})
  list(JOIN ${out_var} "|" ${out_var})
  set(${out_var} "^(${${out_var}})=([0-9A-Fa-f]+)$")
endmacro()


function(_ep_parse_arguments_to_vars
  f
  keywords
  name
  ns
  args
)
  # Transfer the arguments into variables in the calling scope.
  # Because some keywords can be repeated, we can't use cmake_parse_arguments().
  # Instead, we loop through the args and consider the namespace starting with
  # an upper-case letter followed by at least two more upper-case letters,
  # numbers or underscores to be keywords.

  foreach(key IN LISTS keywords)
    unset(${ns}${key})
  endforeach()

  set(key)

  foreach(arg IN LISTS args)
    set(is_value 1)

    if(arg MATCHES "^[A-Z][A-Z0-9_][A-Z0-9_]+$" AND
      NOT (("x${arg}x" STREQUAL "x${key}x") AND
    ("x${key}x" STREQUAL "xCOMMANDx")) AND
      NOT arg MATCHES "^(TRUE|FALSE|YES)$")
      if(arg IN_LIST keywords)
        set(is_value 0)
      endif()
    endif()

    if(is_value)
      if(key)
        # Value
        list(APPEND ${ns}${key} "${arg}")
      else()
        # Missing Keyword
        message(AUTHOR_WARNING
          "value '${arg}' with no previous keyword in ${f}"
        )
      endif()
    else()
      set(key "${arg}")
    endif()
  endforeach()

  foreach(key IN LISTS keywords)
    if(DEFINED ${ns}${key})
      set(${ns}${key} "${${ns}${key}}" PARENT_SCOPE)
    else()
      unset(${ns}${key} PARENT_SCOPE)
    endif()
  endforeach()

endfunction()


# NOTE: This cannot be a macro because that will evaluate anything that looks
#       like a CMake variable in any of the args.
function(_ep_parse_arguments
  f
  keywords
  name
  ns
  args
)
  _ep_parse_arguments_to_vars(
    "${f}"
    "${keywords}"
    ${name}
    ${ns}
    "${args}"
  )

  foreach(key IN LISTS keywords)
    if(DEFINED ${ns}${key})
      set(${ns}${key} "${${ns}${key}}" PARENT_SCOPE)
    else()
      unset(${ns}${key} PARENT_SCOPE)
    endif()
  endforeach()

  # Transfer the arguments to the target as target properties. These are
  # read by the various steps, potentially from different scopes.
  foreach(key IN LISTS keywords)
    if(DEFINED ${ns}${key})
      set_property(TARGET ${name} PROPERTY ${ns}${key} "${${ns}${key}}")
    endif()
  endforeach()

endfunction()


function(_ep_get_tls_version name tls_version_var)
  # Note that the arguments are assumed to have already been parsed and have
  # been translated into variables with the prefix _EP_... by a call to
  # ep_parse_arguments() or ep_parse_arguments_to_vars().
  set(tls_version_regex "^1\\.[0-3]$")
  set(tls_version "${_EP_TLS_VERSION}")
  if(NOT "x${tls_version}" STREQUAL "x")
    if(NOT tls_version MATCHES "${tls_version_regex}")
      message(FATAL_ERROR "TLS_VERSION '${tls_version}' not known")
    endif()
  elseif(NOT "x${CMAKE_TLS_VERSION}" STREQUAL "x")
    set(tls_version "${CMAKE_TLS_VERSION}")
    if(NOT tls_version MATCHES "${tls_version_regex}")
      message(FATAL_ERROR "CMAKE_TLS_VERSION '${tls_version}' not known")
    endif()
  elseif(NOT "x$ENV{CMAKE_TLS_VERSION}" STREQUAL "x")
    set(tls_version "$ENV{CMAKE_TLS_VERSION}")
    if(NOT tls_version MATCHES "${tls_version_regex}")
      message(FATAL_ERROR "ENV{CMAKE_TLS_VERSION} '${tls_version}' not known")
    endif()
  endif()
  set("${tls_version_var}" "${tls_version}" PARENT_SCOPE)
endfunction()


function(_ep_get_tls_verify name tls_verify_var)
  # Note that the arguments are assumed to have already been parsed and have
  # been translated into variables with the prefix _EP_... by a call to
  # ep_parse_arguments() or ep_parse_arguments_to_vars().
  set(tls_verify "${_EP_TLS_VERIFY}")
  if("x${tls_verify}" STREQUAL "x")
    if(NOT "x${CMAKE_TLS_VERIFY}" STREQUAL "x")
      set(tls_verify "${CMAKE_TLS_VERIFY}")
    elseif(NOT "x$ENV{CMAKE_TLS_VERIFY}" STREQUAL "x")
      set(tls_verify "$ENV{CMAKE_TLS_VERIFY}")
    endif()
  endif()
  set("${tls_verify_var}" "${tls_verify}" PARENT_SCOPE)
endfunction()


function(_ep_get_tls_cainfo name tls_cainfo_var)
  # Note that the arguments are assumed to have already been parsed and have
  # been translated into variables with the prefix _EP_... by a call to
  # ep_parse_arguments() or ep_parse_arguments_to_vars().
  set(tls_cainfo "${_EP_TLS_CAINFO}")
  if("x${tls_cainfo}" STREQUAL "x" AND DEFINED CMAKE_TLS_CAINFO)
    set(tls_cainfo "${CMAKE_TLS_CAINFO}")
  endif()
  set("${tls_cainfo_var}" "${tls_cainfo}" PARENT_SCOPE)
endfunction()


function(_ep_get_netrc name netrc_var)
  # Note that the arguments are assumed to have already been parsed and have
  # been translated into variables with the prefix _EP_... by a call to
  # ep_parse_arguments() or ep_parse_arguments_to_vars().
  set(netrc "${_EP_NETRC}")
  if("x${netrc}" STREQUAL "x" AND DEFINED CMAKE_NETRC)
    set(netrc "${CMAKE_NETRC}")
  endif()
  set("${netrc_var}" "${netrc}" PARENT_SCOPE)
endfunction()


function(_ep_get_netrc_file name netrc_file_var)
  # Note that the arguments are assumed to have already been parsed and have
  # been translated into variables with the prefix _EP_... by a call to
  # ep_parse_arguments() or ep_parse_arguments_to_vars().
  set(netrc_file "${_EP_NETRC_FILE}")
  if("x${netrc_file}" STREQUAL "x" AND DEFINED CMAKE_NETRC_FILE)
    set(netrc_file "${CMAKE_NETRC_FILE}")
  endif()
  set("${netrc_file_var}" "${netrc_file}" PARENT_SCOPE)
endfunction()


function(_ep_write_gitclone_script
  script_filename
  source_dir
  git_EXECUTABLE
  git_repository
  git_tag
  git_remote_name
  init_submodules
  git_submodules_recurse
  git_submodules
  git_shallow
  git_progress
  git_config
  src_name
  work_dir
  gitclone_infofile
  gitclone_stampfile
  tls_version
  tls_verify
)

  if(NOT Git_VERSION VERSION_LESS 1.8.5)
    # Use `git checkout <tree-ish> --` to avoid ambiguity with a local path.
    set(git_checkout_explicit-- "--")
  else()
    # Use `git checkout <branch>` even though this risks ambiguity with a
    # local path.  Unfortunately we cannot use `git checkout <tree-ish> --`
    # because that will not search for remote branch names, a common use case.
    set(git_checkout_explicit-- "")
  endif()
  if("${git_tag}" STREQUAL "")
    message(FATAL_ERROR "Tag for git checkout should not be empty.")
  endif()

  if(Git_VERSION VERSION_LESS 2.20 OR
    2.21 VERSION_LESS_EQUAL Git_VERSION)
    set(git_clone_options "--no-checkout")
  else()
    set(git_clone_options)
  endif()
  if(git_shallow)
    if(NOT Git_VERSION VERSION_LESS 1.7.10)
      list(APPEND git_clone_options "--depth 1 --no-single-branch")
    else()
      list(APPEND git_clone_options "--depth 1")
    endif()
  endif()
  if(git_progress)
    list(APPEND git_clone_options --progress)
  endif()
  foreach(config IN LISTS git_config)
    list(APPEND git_clone_options --config \"${config}\")
  endforeach()
  if(NOT ${git_remote_name} STREQUAL "origin")
    list(APPEND git_clone_options --origin \"${git_remote_name}\")
  endif()

  # The clone config option is sticky, it will apply to all subsequent git
  # update operations. The submodules config option is not sticky, because
  # git doesn't provide any way to do that. Thus, we will have to pass the
  # same config option in the update step too for submodules, but not for
  # the main git repo.
  set(git_submodules_config_options "")
  if(NOT "x${tls_version}" STREQUAL "x")
    list(APPEND git_clone_options -c http.sslVersion=tlsv${tls_version})
    list(APPEND git_submodules_config_options -c http.sslVersion=tlsv${tls_version})
  endif()
  if(NOT "x${tls_verify}" STREQUAL "x")
    if(tls_verify)
      # Default git behavior is "true", but the user might have changed the
      # global default to "false". Since TLS_VERIFY was given, ensure we honor
      # the specified setting regardless of what the global default might be.
      list(APPEND git_clone_options -c http.sslVerify=true)
      list(APPEND git_submodules_config_options -c http.sslVerify=true)
    else()
      list(APPEND git_clone_options -c http.sslVerify=false)
      list(APPEND git_submodules_config_options -c http.sslVerify=false)
    endif()
  endif()

  string (REPLACE ";" " " git_clone_options "${git_clone_options}")

  configure_file(
    ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/gitclone.cmake.in
    ${script_filename}
    @ONLY
  )
endfunction()


function(_ep_write_hgclone_script
  script_filename
  source_dir
  hg_EXECUTABLE
  hg_repository
  hg_tag
  src_name
  work_dir
  hgclone_infofile
  hgclone_stampfile
)

  if("${hg_tag}" STREQUAL "")
    message(FATAL_ERROR "Tag for hg checkout should not be empty.")
  endif()

  configure_file(
    ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/hgclone.cmake.in
    ${script_filename}
    @ONLY
  )
endfunction()


function(_ep_write_gitupdate_script
  script_filename
  git_EXECUTABLE
  git_tag
  git_remote_name
  init_submodules
  git_submodules_recurse
  git_submodules
  git_repository
  work_dir
  git_update_strategy
  tls_version
  tls_verify
)

  if("${git_tag}" STREQUAL "")
    message(FATAL_ERROR "Tag for git checkout should not be empty.")
  endif()
  set(git_stash_save_options --quiet)
  if(Git_VERSION VERSION_GREATER_EQUAL 1.7.7)
    # This avoids stashing files covered by .gitignore
    list(APPEND git_stash_save_options --include-untracked)
  elseif(Git_VERSION VERSION_GREATER_EQUAL 1.7.6)
    # Untracked files, but also ignored files, so potentially slower
    list(APPEND git_stash_save_options --all)
  endif()

  # The submodules config option is not sticky, git doesn't provide any way
  # to do that. We have to pass this config option for the update step too.
  # We don't need to set it for the non-submodule update because it gets
  # recorded as part of the clone operation in a sticky manner.
  set(git_submodules_config_options "")
  if(NOT "x${tls_version}" STREQUAL "x")
    list(APPEND git_submodules_config_options -c http.sslVersion=tlsv${tls_version})
  endif()
  if(NOT "x${tls_verify}" STREQUAL "x")
    if(tls_verify)
      # Default git behavior is "true", but the user might have changed the
      # global default to "false". Since TLS_VERIFY was given, ensure we honor
      # the specified setting regardless of what the global default might be.
      list(APPEND git_submodules_config_options -c http.sslVerify=true)
    else()
      list(APPEND git_submodules_config_options -c http.sslVerify=false)
    endif()
  endif()

  configure_file(
    "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/gitupdate.cmake.in"
    "${script_filename}"
    @ONLY
  )
endfunction()


function(_ep_write_downloadfile_script
  script_filename
  REMOTE
  LOCAL
  timeout
  inactivity_timeout
  no_progress
  hash
  tls_version
  tls_verify
  tls_cainfo
  userpwd
  http_headers
  netrc
  netrc_file
)
  if("x${REMOTE}" STREQUAL "x")
    message(FATAL_ERROR "REMOTE can't be empty")
  endif()
  if("x${LOCAL}" STREQUAL "x")
    message(FATAL_ERROR "LOCAL can't be empty")
  endif()

  # REMOTE could contain special characters that parse as separate arguments.
  # Things like parentheses are legitimate characters in a URL, but would be
  # seen as the start of a new unquoted argument by the cmake language parser.
  # Avoid those special cases by preparing quoted strings for direct inclusion
  # in the foreach() call that iterates over the set of URLs in REMOTE.
  set(REMOTE "[====[${REMOTE}]====]")
  string(REPLACE ";" "]====] [====[" REMOTE "${REMOTE}")

  if(timeout)
    set(TIMEOUT_ARGS TIMEOUT ${timeout})
    set(TIMEOUT_MSG "${timeout} seconds")
  else()
    set(TIMEOUT_ARGS "# no TIMEOUT")
    set(TIMEOUT_MSG "none")
  endif()
  if(inactivity_timeout)
    set(INACTIVITY_TIMEOUT_ARGS INACTIVITY_TIMEOUT ${inactivity_timeout})
    set(INACTIVITY_TIMEOUT_MSG "${inactivity_timeout} seconds")
  else()
    set(INACTIVITY_TIMEOUT_ARGS "# no INACTIVITY_TIMEOUT")
    set(INACTIVITY_TIMEOUT_MSG "none")
  endif()

  if(no_progress)
    set(SHOW_PROGRESS "")
  else()
    set(SHOW_PROGRESS "SHOW_PROGRESS")
  endif()

  _ep_get_hash_regex(_ep_hash_regex)
  if("${hash}" MATCHES "${_ep_hash_regex}")
    set(ALGO "${CMAKE_MATCH_1}")
    string(TOLOWER "${CMAKE_MATCH_2}" EXPECT_VALUE)
  else()
    set(ALGO "")
    set(EXPECT_VALUE "")
  endif()

  set(TLS_VERSION_CODE "")
  if(NOT "x${tls_version}" STREQUAL "x")
    set(TLS_VERSION_CODE "set(CMAKE_TLS_VERSION \"${tls_version}\")")
  endif()

  set(TLS_VERIFY_CODE "")
  if(NOT "x${tls_verify}" STREQUAL "x")
    set(TLS_VERIFY_CODE "set(CMAKE_TLS_VERIFY \"${tls_verify}\")")
  endif()

  set(TLS_CAINFO_CODE "")
  if(NOT "x${tls_cainfo}" STREQUAL "x")
    set(TLS_CAINFO_CODE "set(CMAKE_TLS_CAINFO \"${tls_cainfo}\")")
  endif()

  set(NETRC_CODE "")
  if(NOT "x${netrc}" STREQUAL "x")
    set(NETRC_CODE "set(CMAKE_NETRC \"${netrc}\")")
  endif()

  set(NETRC_FILE_CODE "")
  if(NOT "x${netrc_file}" STREQUAL "x")
    set(NETRC_FILE_CODE "set(CMAKE_NETRC_FILE \"${netrc_file}\")")
  endif()

  if(userpwd STREQUAL ":")
    set(USERPWD_ARGS)
  else()
    set(USERPWD_ARGS USERPWD "${userpwd}")
  endif()

  set(HTTP_HEADERS_ARGS "")
  if(NOT http_headers STREQUAL "")
    foreach(header IN LISTS http_headers)
      string(PREPEND HTTP_HEADERS_ARGS
        "HTTPHEADER \"${header}\"\n        "
      )
    endforeach()
  endif()

  # Used variables:
  # * TLS_VERSION_CODE
  # * TLS_VERIFY_CODE
  # * TLS_CAINFO_CODE
  # * ALGO
  # * EXPECT_VALUE
  # * REMOTE
  # * LOCAL
  # * SHOW_PROGRESS
  # * TIMEOUT_ARGS
  # * TIMEOUT_MSG
  # * USERPWD_ARGS
  # * HTTP_HEADERS_ARGS
  configure_file(
    "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/download.cmake.in"
    "${script_filename}"
    @ONLY
  )
endfunction()


function(_ep_write_verifyfile_script
  script_filename
  LOCAL
  hash
)
  _ep_get_hash_regex(_ep_hash_regex)
  if("${hash}" MATCHES "${_ep_hash_regex}")
    set(ALGO "${CMAKE_MATCH_1}")
    string(TOLOWER "${CMAKE_MATCH_2}" EXPECT_VALUE)
  else()
    set(ALGO "")
    set(EXPECT_VALUE "")
  endif()

  # Used variables:
  # * ALGO
  # * EXPECT_VALUE
  # * LOCAL
  configure_file(
    "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/verify.cmake.in"
    "${script_filename}"
    @ONLY
  )
endfunction()


function(_ep_write_extractfile_script
  script_filename
  name
  filename
  directory
  options
)
  # cmake -E tar auto detects the type of archive being extracted
  set(args "xf")

  configure_file(
    "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/extractfile.cmake.in"
    "${script_filename}"
    @ONLY
  )
endfunction()


function(_ep_is_dir_empty dir empty_var)
  file(GLOB gr "${dir}/*")
  if("${gr}" STREQUAL "")
    set(${empty_var} 1 PARENT_SCOPE)
  else()
    set(${empty_var} 0 PARENT_SCOPE)
  endif()
endfunction()

function(_ep_get_git_submodules_recurse git_submodules_recurse)
  # Checks for GIT_SUBMODULES_RECURSE argument. Default is ON, which sets
  # git_submodules_recurse output variable to "--recursive". Otherwise, the
  # output variable is set to an empty value "".
  # Note that the arguments are assumed to have already been parsed and have
  # been translated into variables with the prefix _EP_... by a call to
  # ep_parse_arguments() or ep_parse_arguments_to_vars().
  if(NOT DEFINED _EP_GIT_SUBMODULES_RECURSE)
    set(recurseFlag "--recursive")
  else()
    if(_EP_GIT_SUBMODULES_RECURSE)
      set(recurseFlag "--recursive")
    else()
      set(recurseFlag "")
    endif()
  endif()
  set(${git_submodules_recurse} "${recurseFlag}" PARENT_SCOPE)

  # The git submodule update '--recursive' flag requires git >= v1.6.5
  if(recurseFlag AND Git_VERSION VERSION_LESS 1.6.5)
    message(FATAL_ERROR
      "git version 1.6.5 or later required for --recursive flag with "
      "'git submodule ...': Git_VERSION='${Git_VERSION}'"
    )
  endif()
endfunction()


function(_ep_add_script_commands script_var work_dir cmd)
  # We only support a subset of what ep_replace_location_tags() handles
  set(location_tags
    SOURCE_DIR
    SOURCE_SUBDIR
    BINARY_DIR
    TMP_DIR
    DOWNLOAD_DIR
    DOWNLOADED_FILE
  )

  # There can be multiple COMMANDs, but we have to split those up to
  # one command per call to execute_process()
  string(CONCAT execute_process_cmd
    "execute_process(\n"
    "  WORKING_DIRECTORY \"${work_dir}\"\n"
    "  COMMAND_ERROR_IS_FATAL LAST\n"
  )
  cmake_language(GET_MESSAGE_LOG_LEVEL active_log_level)
  if(active_log_level MATCHES "VERBOSE|DEBUG|TRACE")
    string(APPEND execute_process_cmd "  COMMAND_ECHO STDOUT\n")
  endif()
  string(APPEND execute_process_cmd "  COMMAND ")

  string(APPEND ${script_var} "${execute_process_cmd}")

  foreach(cmd_arg IN LISTS cmd)
    if(cmd_arg STREQUAL "COMMAND")
      string(APPEND ${script_var} "\n)\n${execute_process_cmd}")
    else()
      if(_EP_LIST_SEPARATOR)
        string(REPLACE "${_EP_LIST_SEPARATOR}" "\\;" cmd_arg "${cmd_arg}")
      endif()
      foreach(dir IN LISTS location_tags)
        string(REPLACE "<${dir}>" "${_EP_${dir}}" cmd_arg "${cmd_arg}")
      endforeach()
      string(APPEND ${script_var} " [====[${cmd_arg}]====]")
    endif()
  endforeach()

  string(APPEND ${script_var} "\n)")
  set(${script_var} "${${script_var}}" PARENT_SCOPE)
endfunction()


function(_ep_add_download_command name)
  set(noValueOptions )
  set(singleValueOptions
    SCRIPT_FILE        # These should only be used by FetchContent
    DEPENDS_VARIABLE   #
  )
  set(multiValueOptions )
  cmake_parse_arguments(PARSE_ARGV 1 arg
    "${noValueOptions}" "${singleValueOptions}" "${multiValueOptions}"
  )

  # The various _EP_... variables mentioned here and throughout this function
  # are expected to already have been set by the caller via a call to
  # _ep_parse_arguments() or ep_parse_arguments_to_vars(). Other variables
  # with different names are assigned to for historical reasons only to keep
  # the code more readable and minimize change.

  set(source_dir     "${_EP_SOURCE_DIR}")
  set(stamp_dir      "${_EP_STAMP_DIR}")
  set(download_dir   "${_EP_DOWNLOAD_DIR}")
  set(tmp_dir        "${_EP_TMP_DIR}")

  set(cmd            "${_EP_DOWNLOAD_COMMAND}")
  set(cvs_repository "${_EP_CVS_REPOSITORY}")
  set(svn_repository "${_EP_SVN_REPOSITORY}")
  set(git_repository "${_EP_GIT_REPOSITORY}")
  set(hg_repository  "${_EP_HG_REPOSITORY}")
  set(url            "${_EP_URL}")
  set(fname          "${_EP_DOWNLOAD_NAME}")

  # TODO: Perhaps file:// should be copied to download dir before extraction.
  string(REGEX REPLACE "file://" "" url "${url}")

  set(step_script_contents)
  set(depends)
  set(comment)
  set(work_dir)
  set(extra_repo_info)

  if(DEFINED _EP_DOWNLOAD_COMMAND)
    set(work_dir ${download_dir})
    set(method custom)
    if(NOT "x${cmd}" STREQUAL "x" AND arg_SCRIPT_FILE)
      _ep_add_script_commands(
        step_script_contents
        "${work_dir}"
        "${cmd}"   # Must be a single quoted argument
      )
    endif()

  elseif(cvs_repository)
    set(method cvs)
    find_package(CVS QUIET)
    if(NOT CVS_EXECUTABLE)
      message(FATAL_ERROR "error: could not find cvs for checkout of ${name}")
    endif()

    set(cvs_module "${_EP_CVS_MODULE}")
    if(NOT cvs_module)
      message(FATAL_ERROR "error: no CVS_MODULE")
    endif()

    set(cvs_tag "${_EP_CVS_TAG}")
    get_filename_component(src_name "${source_dir}" NAME)
    get_filename_component(work_dir "${source_dir}" PATH)
    set(comment "Performing download step (CVS checkout) for '${name}'")
    set(cmd
      ${CVS_EXECUTABLE}
      -d ${cvs_repository}
      -q
      co ${cvs_tag}
      -d ${src_name}
      ${cvs_module}
    )
    if(arg_SCRIPT_FILE)
      _ep_add_script_commands(
        step_script_contents
        "${work_dir}"
        "${cmd}"   # Must be a single quoted argument
      )
    endif()

  elseif(svn_repository)
    set(method svn)
    find_package(Subversion QUIET)
    if(NOT Subversion_SVN_EXECUTABLE)
      message(FATAL_ERROR "error: could not find svn for checkout of ${name}")
    endif()

    set(svn_trust_cert "${_EP_SVN_TRUST_CERT}")
    set(uses_terminal  "${_EP_USES_TERMINAL_DOWNLOAD}")

    get_filename_component(src_name "${source_dir}" NAME)
    get_filename_component(work_dir "${source_dir}" PATH)
    set(comment "Performing download step (SVN checkout) for '${name}'")
    set(cmd
      ${Subversion_SVN_EXECUTABLE}
      co
      ${svn_repository}
      ${_EP_SVN_REVISION}
    )
    # The --trust-server-cert option requires --non-interactive
    if(svn_trust_cert OR NOT uses_terminal)
      list(APPEND cmd "--non-interactive")
    endif()
    if(svn_trust_cert)
      list(APPEND cmd "--trust-server-cert")
    endif()
    if(DEFINED _EP_SVN_USERNAME)
      list(APPEND cmd "--username=${_EP_SVN_USERNAME}")
    endif()
    if(DEFINED _EP_SVN_PASSWORD)
      list(APPEND cmd "--password=${_EP_SVN_PASSWORD}")
    endif()
    list(APPEND cmd ${src_name})

    if(arg_SCRIPT_FILE)
      _ep_add_script_commands(
        step_script_contents
        "${work_dir}"
        "${cmd}"   # Must be a single quoted argument
      )
    endif()

  elseif(git_repository)
    set(method git)
    # FetchContent gives us these directly, so don't try to recompute them
    if(NOT GIT_EXECUTABLE OR NOT Git_VERSION)
      unset(CMAKE_MODULE_PATH) # Use CMake builtin find module
      find_package(Git QUIET)
      if(NOT GIT_EXECUTABLE)
        message(FATAL_ERROR "error: could not find git for clone of ${name}")
      endif()
    endif()

    _ep_get_git_submodules_recurse(git_submodules_recurse)

    set(git_tag "${_EP_GIT_TAG}")
    if(NOT git_tag)
      set(git_tag "master")
    endif()

    set(git_init_submodules TRUE)
    if(DEFINED _EP_GIT_SUBMODULES)
      set(git_submodules "${_EP_GIT_SUBMODULES}")
      if(git_submodules STREQUAL "" AND _EP_CMP0097 STREQUAL "NEW")
        set(git_init_submodules FALSE)
      endif()
    endif()

    set(git_remote_name "${_EP_GIT_REMOTE_NAME}")
    if(NOT git_remote_name)
      set(git_remote_name "origin")
    endif()

    _ep_get_tls_version(${name} tls_version)
    _ep_get_tls_verify(${name} tls_verify)
    set(git_shallow  "${_EP_GIT_SHALLOW}")
    set(git_progress "${_EP_GIT_PROGRESS}")
    set(git_config   "${_EP_GIT_CONFIG}")

    # If git supports it, make checkouts quiet when checking out a git hash.
    # This avoids the very noisy detached head message.
    if(Git_VERSION VERSION_GREATER_EQUAL 1.7.7)
      list(PREPEND git_config advice.detachedHead=false)
    endif()

    # The command doesn't expose any details, so we need to record additional
    # information in the RepositoryInfo.txt file. For the download step, only
    # the things specifically affecting the clone operation should be recorded.
    # If the repo changes, the clone script should be run again.
    # But if only the tag changes, avoid running the clone script again.
    # Let the 'always' running update step checkout the new tag.
    #
    set(extra_repo_info
      "repository=${git_repository}
remote=${git_remote_name}
init_submodules=${git_init_submodules}
recurse_submodules=${git_submodules_recurse}
submodules=${git_submodules}
CMP0097=${_EP_CMP0097}
      ")
    get_filename_component(src_name "${source_dir}" NAME)
    get_filename_component(work_dir "${source_dir}" PATH)

    # Since git clone doesn't succeed if the non-empty source_dir exists,
    # create a cmake script to invoke as download command.
    # The script will delete the source directory and then call git clone.
    #
    set(clone_script ${tmp_dir}/${name}-gitclone.cmake)
    _ep_write_gitclone_script(
      ${clone_script}
      ${source_dir}
      ${GIT_EXECUTABLE}
      ${git_repository}
      ${git_tag}
      ${git_remote_name}
      ${git_init_submodules}
      "${git_submodules_recurse}"
      "${git_submodules}"
      "${git_shallow}"
      "${git_progress}"
      "${git_config}"
      ${src_name}
      ${work_dir}
      ${stamp_dir}/${name}-gitinfo.txt
      ${stamp_dir}/${name}-gitclone-lastrun.txt
      "${tls_version}"
      "${tls_verify}"
    )
    set(comment "Performing download step (git clone) for '${name}'")
    set(cmd ${CMAKE_COMMAND}
      -DCMAKE_MESSAGE_LOG_LEVEL=VERBOSE
      -P ${clone_script}
    )

    if(arg_SCRIPT_FILE)
      set(step_script_contents "include(\"${clone_script}\")")
      list(APPEND depends ${clone_script})
    endif()

  elseif(hg_repository)
    set(method hg)
    find_package(Hg QUIET)
    if(NOT HG_EXECUTABLE)
      message(FATAL_ERROR "error: could not find hg for clone of ${name}")
    endif()

    set(hg_tag "${_EP_HG_TAG}")
    if(NOT hg_tag)
      set(hg_tag "tip")
    endif()

    # The command doesn't expose any details, so we need to record additional
    # information in the RepositoryInfo.txt file. For the download step, only
    # the things specifically affecting the clone operation should be recorded.
    # If the repo changes, the clone script should be run again.
    # But if only the tag changes, avoid running the clone script again.
    # Let the 'always' running update step checkout the new tag.
    #
    set(extra_repo_info "repository=${hg_repository}")
    get_filename_component(src_name "${source_dir}" NAME)
    get_filename_component(work_dir "${source_dir}" PATH)

    # Since hg clone doesn't succeed if the non-empty source_dir exists,
    # create a cmake script to invoke as download command.
    # The script will delete the source directory and then call hg clone.
    #
    set(clone_script ${tmp_dir}/${name}-hgclone.cmake)
    _ep_write_hgclone_script(
      ${clone_script}
      ${source_dir}
      ${HG_EXECUTABLE}
      ${hg_repository}
      ${hg_tag}
      ${src_name}
      ${work_dir}
      ${stamp_dir}/${name}-hginfo.txt
      ${stamp_dir}/${name}-hgclone-lastrun.txt
    )
    set(comment "Performing download step (hg clone) for '${name}'")
    set(cmd ${CMAKE_COMMAND}
      -DCMAKE_MESSAGE_LOG_LEVEL=VERBOSE
      -P ${clone_script}
    )

    if(arg_SCRIPT_FILE)
      set(step_script_contents "include(\"${clone_script}\")")
      list(APPEND depends ${clone_script})
    endif()

  elseif(url)
    set(method url)
    get_filename_component(work_dir "${source_dir}" PATH)
    set(hash "${_EP_URL_HASH}")
    _ep_get_hash_regex(_ep_hash_regex)
    if(hash AND NOT "${hash}" MATCHES "${_ep_hash_regex}")
      _ep_get_hash_algos(_ep_hash_algos)
      list(JOIN _ep_hash_algos "|" _ep_hash_algos)
      message(FATAL_ERROR
        "URL_HASH is set to\n"
        "  ${hash}\n"
        "but must be ALGO=value where ALGO is\n"
        "  ${_ep_hash_algos}\n"
        "and value is a hex string."
      )
    endif()
    set(md5 "${_EP_URL_MD5}")
    if(md5 AND NOT "MD5=${md5}" MATCHES "${_ep_hash_regex}")
      message(FATAL_ERROR
        "URL_MD5 is set to\n"
        "  ${md5}\n"
        "but must be a hex string."
      )
    endif()
    if(md5 AND NOT hash)
      set(hash "MD5=${md5}")
    endif()
    set(extra_repo_info
      "url(s)=${url}
hash=${hash}
      ")

    list(LENGTH url url_list_length)
    if(NOT "${url_list_length}" STREQUAL "1")
      foreach(entry IN LISTS url)
        if(NOT "${entry}" MATCHES "^[a-z]+://")
          message(FATAL_ERROR
            "At least one entry of URL is a path (invalid in a list)"
          )
        endif()
      endforeach()
      if("x${fname}" STREQUAL "x")
        list(GET url 0 fname)
      endif()
    endif()

    if(IS_DIRECTORY "${url}")
      get_filename_component(abs_dir "${url}" ABSOLUTE)
      set(comment "Performing download step (DIR copy) for '${name}'")
      set(cmd
        ${CMAKE_COMMAND} -E rm -rf ${source_dir}
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${abs_dir} ${source_dir}
      )
      if(arg_SCRIPT_FILE)
        # While it may be tempting to implement the two operations directly
        # with file(), the behavior is different. file(COPY) preserves input
        # file timestamps, which we don't want. Therefore, still use the same
        # external commands so that we get the same behavior.
        _ep_add_script_commands(
          step_script_contents
          "${work_dir}"
          "${cmd}"   # Must be a single quoted argument
        )
      endif()
    else()
      set(no_extract "${_EP_DOWNLOAD_NO_EXTRACT}")
      string(APPEND extra_repo_info "no_extract=${no_extract}\n")
      set(verify_script "${stamp_dir}/verify-${name}.cmake")
      if("${url}" MATCHES "^[a-z]+://")
        # TODO: Should download and extraction be different steps?
        if("x${fname}" STREQUAL "x")
          set(fname "${url}")
        endif()
        set(ext_regex [[7z|tar|tar\.bz2|tar\.gz|tar\.xz|tbz2|tgz|txz|zip]])
        if("${fname}" MATCHES "([^/\\?#]+(\\.|=)(${ext_regex}))([/?#].*)?$")
          set(fname "${CMAKE_MATCH_1}")
        elseif(no_extract)
          get_filename_component(fname "${fname}" NAME)
        else()
          # Fall back to a default file name.  The actual file name does not
          # matter because it is used only internally and our extraction tool
          # inspects the file content directly.  If it turns out the wrong URL
          # was given that will be revealed during the build which is an easier
          # place for users to diagnose than an error here anyway.
          set(fname "archive.tar")
        endif()
        string(REPLACE ";" "-" fname "${fname}")
        set(file ${download_dir}/${fname})
        set(timeout "${_EP_TIMEOUT}")
        set(inactivity_timeout "${_EP_INACTIVITY_TIMEOUT}")
        set(no_progress "${_EP_DOWNLOAD_NO_PROGRESS}")
        _ep_get_tls_version(${name} tls_version)
        _ep_get_tls_verify(${name} tls_verify)
        _ep_get_tls_cainfo(${name} tls_cainfo)
        _ep_get_netrc(${name} netrc)
        _ep_get_netrc_file(${name} netrc_file)
        set(http_username "${_EP_HTTP_USERNAME}")
        set(http_password "${_EP_HTTP_PASSWORD}")
        set(http_headers  "${_EP_HTTP_HEADER}")
        set(download_script "${stamp_dir}/download-${name}.cmake")
        _ep_write_downloadfile_script(
          "${download_script}"
          "${url}"
          "${file}"
          "${timeout}"
          "${inactivity_timeout}"
          "${no_progress}"
          "${hash}"
          "${tls_version}"
          "${tls_verify}"
          "${tls_cainfo}"
          "${http_username}:${http_password}"
          "${http_headers}"
          "${netrc}"
          "${netrc_file}"
        )
        set(cmd
          ${CMAKE_COMMAND}
            -DCMAKE_MESSAGE_LOG_LEVEL=VERBOSE
            -P "${download_script}"
          COMMAND
        )
        if(arg_SCRIPT_FILE)
          set(step_script_contents "include(\"${download_script}\")\n")
        endif()

        if (no_extract)
          set(steps "download and verify")
        else ()
          set(steps "download, verify and extract")
        endif ()
        set(comment "Performing download step (${steps}) for '${name}'")
        # already verified by 'download_script'
        # We use file(CONFIGURE) instead of file(WRITE) to avoid updating the
        # timestamp when the file already existed and was empty.
        file(CONFIGURE OUTPUT "${verify_script}" CONTENT "")

        # Rather than adding everything to the RepositoryInfo.txt file, it is
        # more robust to just depend on the download script. That way, we will
        # re-download if any aspect of the download changes.
        list(APPEND depends "${download_script}")
      else()
        set(file "${url}")
        if (no_extract)
          set(steps "verify")
        else ()
          set(steps "verify and extract")
        endif ()
        set(comment "Performing download step (${steps}) for '${name}'")
        _ep_write_verifyfile_script(
          "${verify_script}"
          "${file}"
          "${hash}"
        )
      endif()
      list(APPEND cmd ${CMAKE_COMMAND}
        -DCMAKE_MESSAGE_LOG_LEVEL=VERBOSE
        -P ${verify_script}
      )
      if(arg_SCRIPT_FILE)
        string(APPEND step_script_contents "include(\"${verify_script}\")\n")
        list(APPEND depends ${verify_script})
      endif()
      set(extract_timestamp "${_EP_DOWNLOAD_EXTRACT_TIMESTAMP}")
      if(no_extract)
        if(DEFINED _EP_DOWNLOAD_EXTRACT_TIMESTAMP)
          message(FATAL_ERROR
            "Cannot specify DOWNLOAD_EXTRACT_TIMESTAMP when using "
            "DOWNLOAD_NO_EXTRACT TRUE"
          )
        endif()
        if(arg_SCRIPT_FILE)
          # There's no target to record the location of the downloaded file.
          # Instead, we copy it to the source directory within the script,
          # which is what FetchContent always does in this situation.
          cmake_path(SET safe_file NORMALIZE "${file}")
          cmake_path(GET safe_file FILENAME filename)
          string(APPEND step_script_contents
            "file(COPY_FILE\n"
            "  \"${file}\"\n"
            "  \"${source_dir}/${filename}\"\n"
            "  ONLY_IF_DIFFERENT\n"
            "  INPUT_MAY_BE_RECENT\n"
            ")"
          )
          list(APPEND depends ${source_dir}/${filename})
        else()
          set_property(TARGET ${name} PROPERTY _EP_DOWNLOADED_FILE ${file})
        endif()
      else()
        if(NOT DEFINED _EP_DOWNLOAD_EXTRACT_TIMESTAMP)
          # Default depends on policy CMP0135
          if(_EP_CMP0135 STREQUAL "")
            message(AUTHOR_WARNING
              "The DOWNLOAD_EXTRACT_TIMESTAMP option was not given and policy "
              "CMP0135 is not set. The policy's OLD behavior will be used. "
              "When using a URL download, the timestamps of extracted files "
              "should preferably be that of the time of extraction, otherwise "
              "code that depends on the extracted contents might not be "
              "rebuilt if the URL changes. The OLD behavior preserves the "
              "timestamps from the archive instead, but this is usually not "
              "what you want. Update your project to the NEW behavior or "
              "specify the DOWNLOAD_EXTRACT_TIMESTAMP option with a value of "
              "true to avoid this robustness issue."
            )
            set(extract_timestamp TRUE)
          elseif(_EP_CMP0135 STREQUAL "NEW")
            set(extract_timestamp FALSE)
          else()
            set(extract_timestamp TRUE)
          endif()
        endif()
        if(extract_timestamp)
          set(options "")
        else()
          set(options "--touch")
        endif()
        set(extract_script "${stamp_dir}/extract-${name}.cmake")
        _ep_write_extractfile_script(
          "${extract_script}"
          "${name}"
          "${file}"
          "${source_dir}"
          "${options}"
        )
        list(APPEND cmd
          COMMAND ${CMAKE_COMMAND}
            -DCMAKE_MESSAGE_LOG_LEVEL=VERBOSE
            -P ${extract_script}
        )
        if(arg_SCRIPT_FILE)
          string(APPEND step_script_contents "include(\"${extract_script}\")\n")
          list(APPEND depends ${extract_script})
        endif()
      endif ()
    endif()
  else()
    set(method source_dir)
    _ep_is_dir_empty("${source_dir}" empty)
    if(${empty})
      message(FATAL_ERROR
        "No download info given for '${name}' and its source directory:\n"
        " ${source_dir}\n"
        "is not an existing non-empty directory.  Please specify one of:\n"
        " * SOURCE_DIR with an existing non-empty directory\n"
        " * DOWNLOAD_COMMAND\n"
        " * URL\n"
        " * GIT_REPOSITORY\n"
        " * SVN_REPOSITORY\n"
        " * HG_REPOSITORY\n"
        " * CVS_REPOSITORY and CVS_MODULE"
      )
    endif()
    if(arg_SCRIPT_FILE)
      set(step_script_contents "message(VERBOSE [[Using SOURCE_DIR as is]])")
    endif()
  endif()

  # We use configure_file() to write the repo_info_file so that the file's
  # timestamp is not updated if we don't change the contents

  set(repo_info_file ${stamp_dir}/${name}-${method}info.txt)
  list(APPEND depends ${repo_info_file})
  configure_file(
    "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/RepositoryInfo.txt.in"
    "${repo_info_file}"
    @ONLY
  )

  if(arg_SCRIPT_FILE)
    set(step_name download)
    configure_file(
      "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/stepscript.cmake.in"
      "${arg_SCRIPT_FILE}"
      @ONLY
    )
    set(${arg_DEPENDS_VARIABLE} "${depends}" PARENT_SCOPE)
    return()
  endif()

  # Nothing below this point is applicable when we've been asked to put the
  # download step in a script file (which is the FetchContent case).

  if(_EP_LOG_DOWNLOAD)
    set(log LOG 1)
  else()
    set(log "")
  endif()

  if(_EP_USES_TERMINAL_DOWNLOAD)
    set(uses_terminal USES_TERMINAL 1)
  else()
    set(uses_terminal "")
  endif()

  set(__cmdQuoted)
  foreach(__item IN LISTS cmd)
    string(APPEND __cmdQuoted " [==[${__item}]==]")
  endforeach()
  cmake_language(EVAL CODE "
    ExternalProject_Add_Step(\${name} download
      INDEPENDENT TRUE
      COMMENT \${comment}
      COMMAND ${__cmdQuoted}
      WORKING_DIRECTORY \${work_dir}
      DEPENDS \${depends}
      DEPENDEES mkdir
      ${log}
      ${uses_terminal}
    )"
  )
endfunction()

function(_ep_get_update_disconnected var name)
  # Note that the arguments are assumed to have already been parsed and have
  # been translated into variables with the prefix _EP_... by a call to
  # ep_parse_arguments() or ep_parse_arguments_to_vars().
  if(DEFINED _EP_UPDATE_DISCONNECTED)
    set(update_disconnected "${_EP_UPDATE_DISCONNECTED}")
  else()
    get_property(update_disconnected
      DIRECTORY
      PROPERTY EP_UPDATE_DISCONNECTED
    )
  endif()
  set(${var} "${update_disconnected}" PARENT_SCOPE)
endfunction()

function(_ep_add_update_command name)
  set(noValueOptions )
  set(singleValueOptions
    SCRIPT_FILE       # These should only be used by FetchContent
    DEPEND_VARIABLE   #
  )
  set(multiValueOptions )
  cmake_parse_arguments(PARSE_ARGV 1 arg
    "${noValueOptions}" "${singleValueOptions}" "${multiValueOptions}"
  )

  # The various _EP_... variables mentioned here and throughout this function
  # are expected to already have been set by the caller via a call to
  # _ep_parse_arguments() or ep_parse_arguments_to_vars(). Other variables
  # with different names are assigned to for historical reasons only to keep
  # the code more readable and minimize change.

  set(source_dir     "${_EP_SOURCE_DIR}")
  set(stamp_dir      "${_EP_STAMP_DIR}")
  set(tmp_dir        "${_EP_TMP_DIR}")

  set(cmd            "${_EP_UPDATE_COMMAND}")
  set(cvs_repository "${_EP_CVS_REPOSITORY}")
  set(svn_repository "${_EP_SVN_REPOSITORY}")
  set(git_repository "${_EP_GIT_REPOSITORY}")
  set(hg_repository  "${_EP_HG_REPOSITORY}")

  _ep_get_update_disconnected(update_disconnected ${name})

  set(work_dir)
  set(cmd_disconnected)
  set(comment)
  set(comment_disconnected)
  set(always)
  set(file_deps)

  if(DEFINED _EP_UPDATE_COMMAND)
    set(work_dir ${source_dir})
    if(NOT "x${cmd}" STREQUAL "x")
      set(always 1)
      _ep_add_script_commands(
        step_script_contents
        "${work_dir}"
        "${cmd}"   # Must be a single quoted argument
      )
    endif()

  elseif(cvs_repository)
    if(NOT CVS_EXECUTABLE)
      message(FATAL_ERROR "error: could not find cvs for update of ${name}")
    endif()
    set(work_dir ${source_dir})
    set(comment "Performing update step (CVS update) for '${name}'")
    set(cvs_tag "${_EP_CVS_TAG}")
    set(cmd ${CVS_EXECUTABLE} -d ${cvs_repository} -q up -dP ${cvs_tag})
    set(always 1)

    if(arg_SCRIPT_FILE)
      _ep_add_script_commands(
        step_script_contents
        "${work_dir}"
        "${cmd}"   # Must be a single quoted argument
      )
    endif()

  elseif(svn_repository)
    if(NOT Subversion_SVN_EXECUTABLE)
      message(FATAL_ERROR "error: could not find svn for update of ${name}")
    endif()
    set(work_dir ${source_dir})
    set(comment "Performing update step (SVN update) for '${name}'")
    set(svn_trust_cert "${_EP_SVN_TRUST_CERT}")
    set(uses_terminal  "${_EP_USES_TERMINAL_UPDATE}")
    set(cmd
      ${Subversion_SVN_EXECUTABLE}
      up
      ${_EP_SVN_REVISION}
    )
    # The --trust-server-cert option requires --non-interactive
    if(svn_trust_cert OR NOT uses_terminal)
      list(APPEND cmd "--non-interactive")
    endif()
    if(svn_trust_cert)
      list(APPEND cmd --trust-server-cert)
    endif()
    if(DEFINED _EP_SVN_USERNAME)
      list(APPEND cmd "--username=${_EP_SVN_USERNAME}")
    endif()
    if(DEFINED _EP_SVN_PASSWORD)
      list(APPEND cmd "--password=${_EP_SVN_PASSWORD}")
    endif()
    set(always 1)

    if(arg_SCRIPT_FILE)
      _ep_add_script_commands(
        step_script_contents
        "${work_dir}"
        "${cmd}"   # Must be a single quoted argument
      )
    endif()

  elseif(git_repository)
    # FetchContent gives us these directly, so don't try to recompute them
    if(NOT GIT_EXECUTABLE OR NOT Git_VERSION)
      unset(CMAKE_MODULE_PATH) # Use CMake builtin find module
      find_package(Git QUIET)
      if(NOT GIT_EXECUTABLE)
        message(FATAL_ERROR "error: could not find git for fetch of ${name}")
      endif()
    endif()
    set(work_dir ${source_dir})
    set(comment "Performing update step for '${name}'")
    set(comment_disconnected "Performing disconnected update step for '${name}'")

    if(update_disconnected)
      set(can_fetch_default NO)
    else()
      set(can_fetch_default YES)
    endif()

    set(git_tag "${_EP_GIT_TAG}")
    if(NOT git_tag)
      set(git_tag "master")
    endif()

    set(git_remote_name "${_EP_GIT_REMOTE_NAME}")
    if(NOT git_remote_name)
      set(git_remote_name "origin")
    endif()

    set(git_init_submodules TRUE)
    if(DEFINED _EP_GIT_SUBMODULES)
      set(git_submodules "${_EP_GIT_SUBMODULES}")
      if(git_submodules STREQUAL "" AND _EP_CMP0097 STREQUAL "NEW")
        set(git_init_submodules FALSE)
      endif()
    endif()

    set(git_update_strategy "${_EP_GIT_REMOTE_UPDATE_STRATEGY}")
    if(NOT git_update_strategy)
      set(git_update_strategy "${CMAKE_EP_GIT_REMOTE_UPDATE_STRATEGY}")
    endif()
    if(NOT git_update_strategy)
      set(git_update_strategy REBASE)
    endif()
    set(strategies CHECKOUT REBASE REBASE_CHECKOUT)
    if(NOT git_update_strategy IN_LIST strategies)
      message(FATAL_ERROR
        "'${git_update_strategy}' is not one of the supported strategies: "
        "${strategies}"
      )
    endif()

    _ep_get_git_submodules_recurse(git_submodules_recurse)

    _ep_get_tls_version(${name} tls_version)
    _ep_get_tls_verify(${name} tls_verify)

    set(update_script "${tmp_dir}/${name}-gitupdate.cmake")
    list(APPEND file_deps ${update_script})
    _ep_write_gitupdate_script(
      "${update_script}"
      "${GIT_EXECUTABLE}"
      "${git_tag}"
      "${git_remote_name}"
      "${git_init_submodules}"
      "${git_submodules_recurse}"
      "${git_submodules}"
      "${git_repository}"
      "${work_dir}"
      "${git_update_strategy}"
      "${tls_version}"
      "${tls_verify}"
    )
    set(cmd ${CMAKE_COMMAND}
      -Dcan_fetch=YES
      -DCMAKE_MESSAGE_LOG_LEVEL=VERBOSE
      -P ${update_script}
    )
    set(cmd_disconnected ${CMAKE_COMMAND}
      -Dcan_fetch=NO
      -DCMAKE_MESSAGE_LOG_LEVEL=VERBOSE
      -P ${update_script}
    )
    set(always 1)

    if(arg_SCRIPT_FILE)
      set(step_script_contents "include(\"${update_script}\")")
    endif()

  elseif(hg_repository)
    if(NOT HG_EXECUTABLE)
      message(FATAL_ERROR "error: could not find hg for pull of ${name}")
    endif()
    set(work_dir ${source_dir})
    set(comment "Performing update step (hg pull) for '${name}'")
    set(comment_disconnected "Performing disconnected update step for '${name}'")

    set(hg_tag "${_EP_HG_TAG}")
    if(NOT hg_tag)
      set(hg_tag "tip")
    endif()

    if("${Hg_VERSION}" STREQUAL "2.1")
      set(notesAnchor
        "#A2.1.1:_revert_pull_return_code_change.2C_compile_issue_on_OS_X"
      )
      message(WARNING
        "Mercurial 2.1 does not distinguish an empty pull from a failed pull:
 http://mercurial.selenic.com/wiki/UpgradeNotes${notesAnchor}
 http://thread.gmane.org/gmane.comp.version-control.mercurial.devel/47656
Update to Mercurial >= 2.1.1.
")
    endif()

    set(cmd
      ${HG_EXECUTABLE} pull
      COMMAND ${HG_EXECUTABLE} update ${hg_tag}
    )
    set(cmd_disconnected ${HG_EXECUTABLE} update ${hg_tag})
    set(always 1)

    if(arg_SCRIPT_FILE)
      # These commands are simple, and we know whether updates need to be
      # disconnected or not for this case, so write them directly instead of
      # forming them from "cmd" and "cmd_disconnected".
      if(NOT update_disconnected)
        string(APPEND step_script_contents
          "execute_process(\n"
          "  WORKING_DIRECTORY \"${work_dir}\"\n"
          "  COMMAND_ERROR_IS_FATAL LAST\n"
          "  COMMAND \"${HG_EXECUTABLE}\" pull\n"
          ")"
        )
      endif()
      string(APPEND step_script_contents
        "execute_process(\n"
        "  WORKING_DIRECTORY \"${work_dir}\"\n"
        "  COMMAND_ERROR_IS_FATAL LAST\n"
        "  COMMAND \"${HG_EXECUTABLE}\" update \"${hg_tag}\"\n"
        ")"
      )
    endif()
  endif()

  # We use configure_file() to write the update_info_file so that the file's
  # timestamp is not updated if we don't change the contents
  if(NOT DEFINED cmd_disconnected)
    set(cmd_disconnected "${cmd}")
  endif()
  set(update_info_file ${stamp_dir}/${name}-update-info.txt)
  list(APPEND file_deps ${update_info_file})
  configure_file(
    "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/UpdateInfo.txt.in"
    "${update_info_file}"
    @ONLY
  )

  if(arg_SCRIPT_FILE)
    set(step_name update)
    configure_file(
      "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/stepscript.cmake.in"
      "${arg_SCRIPT_FILE}"
      @ONLY
    )
    set(${arg_DEPENDS_VARIABLE} "${file_deps}" PARENT_SCOPE)
    return()
  endif()

  # Nothing below this point is applicable when we've been asked to put the
  # update step in a script file (which is the FetchContent case).

  if(_EP_LOG_UPDATE)
    set(log LOG 1)
  else()
    set(log "")
  endif()

  if(_EP_USES_TERMINAL_UPDATE)
    set(uses_terminal USES_TERMINAL 1)
  else()
    set(uses_terminal "")
  endif()

  set(__cmdQuoted)
  foreach(__item IN LISTS cmd)
    string(APPEND __cmdQuoted " [==[${__item}]==]")
  endforeach()
  cmake_language(EVAL CODE "
    ExternalProject_Add_Step(${name} update
      INDEPENDENT TRUE
      COMMENT \${comment}
      COMMAND ${__cmdQuoted}
      ALWAYS \${always}
      EXCLUDE_FROM_MAIN \${update_disconnected}
      WORKING_DIRECTORY \${work_dir}
      DEPENDEES download
      DEPENDS \${file_deps}
      ${log}
      ${uses_terminal}
    )"
  )
  if(update_disconnected)
    if(NOT DEFINED comment_disconnected)
      set(comment_disconnected "${comment}")
    endif()
    set(__cmdQuoted)
    foreach(__item IN LISTS cmd_disconnected)
      string(APPEND __cmdQuoted " [==[${__item}]==]")
    endforeach()

    cmake_language(EVAL CODE "
      ExternalProject_Add_Step(${name} update_disconnected
        INDEPENDENT TRUE
        COMMENT \${comment_disconnected}
        COMMAND ${__cmdQuoted}
        WORKING_DIRECTORY \${work_dir}
        DEPENDEES download
        DEPENDS \${file_deps}
        ${log}
        ${uses_terminal}
      )"
    )
  endif()

endfunction()


function(_ep_add_patch_command name)
  set(noValueOptions )
  set(singleValueOptions
    SCRIPT_FILE        # These should only be used by FetchContent
  )
  set(multiValueOptions )
  cmake_parse_arguments(PARSE_ARGV 1 arg
    "${noValueOptions}" "${singleValueOptions}" "${multiValueOptions}"
  )

  # The various _EP_... variables mentioned here and throughout this function
  # are expected to already have been set by the caller via a call to
  # _ep_parse_arguments() or ep_parse_arguments_to_vars(). Other variables
  # with different names are assigned to for historical reasons only to keep
  # the code more readable and minimize change.

  set(source_dir "${_EP_SOURCE_DIR}")
  set(stamp_dir  "${_EP_STAMP_DIR}")

  set(cmd "${_EP_PATCH_COMMAND}")
  set(step_script_contents "")

  set(work_dir)
  if(DEFINED _EP_PATCH_COMMAND)
    set(work_dir ${source_dir})
    if(arg_SCRIPT_FILE)
      _ep_add_script_commands(
        step_script_contents
        "${work_dir}"
        "${cmd}"   # Must be a single quoted argument
      )
    endif()
  endif()

  # We use configure_file() to write the patch_info_file so that the file's
  # timestamp is not updated if we don't change the contents
  set(patch_info_file ${stamp_dir}/${name}-patch-info.txt)
  configure_file(
    "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/PatchInfo.txt.in"
    "${patch_info_file}"
    @ONLY
  )

  if(arg_SCRIPT_FILE)
    set(step_name patch)
    configure_file(
      "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/stepscript.cmake.in"
      "${arg_SCRIPT_FILE}"
      @ONLY
    )
    return()
  endif()

  # Nothing below this point is applicable when we've been asked to put the
  # patch step in a script file (which is the FetchContent case).

  if(_EP_LOG_PATCH)
    set(log LOG 1)
  else()
    set(log "")
  endif()

  if(_EP_USES_TERMINAL_PATCH)
    set(uses_terminal USES_TERMINAL 1)
  else()
    set(uses_terminal "")
  endif()

  _ep_get_update_disconnected(update_disconnected ${name})

  set(__cmdQuoted)
  foreach(__item IN LISTS cmd)
    string(APPEND __cmdQuoted " [==[${__item}]==]")
  endforeach()
  cmake_language(EVAL CODE "
    ExternalProject_Add_Step(${name} patch
      INDEPENDENT TRUE
      COMMAND ${__cmdQuoted}
      WORKING_DIRECTORY \${work_dir}
      EXCLUDE_FROM_MAIN \${update_disconnected}
      DEPENDEES update
      DEPENDS \${patch_info_file}
      ${log}
      ${uses_terminal}
    )"
  )

  if(update_disconnected)
    cmake_language(EVAL CODE "
      ExternalProject_Add_Step(${name} patch_disconnected
        INDEPENDENT TRUE
        COMMAND ${__cmdQuoted}
        WORKING_DIRECTORY \${work_dir}
        DEPENDEES update_disconnected
        DEPENDS \${patch_info_file}
        ${log}
        ${uses_terminal}
      )"
    )
  endif()

endfunction()


macro(_ep_get_add_keywords out_var)
  set(${out_var}
    #
    # Directory options
    #
    PREFIX
    TMP_DIR
    STAMP_DIR
    LOG_DIR
    DOWNLOAD_DIR
    SOURCE_DIR
    BINARY_DIR
    INSTALL_DIR
    #
    # Download step options
    #
    DOWNLOAD_COMMAND
    #
    URL
    URL_HASH
    URL_MD5
    DOWNLOAD_NAME
    DOWNLOAD_EXTRACT_TIMESTAMP
    DOWNLOAD_NO_EXTRACT
    DOWNLOAD_NO_PROGRESS
    TIMEOUT
    INACTIVITY_TIMEOUT
    HTTP_USERNAME
    HTTP_PASSWORD
    HTTP_HEADER
    TLS_VERSION    # Also used for git clone operations
    TLS_VERIFY     # Also used for git clone operations
    TLS_CAINFO
    NETRC
    NETRC_FILE
    #
    GIT_REPOSITORY
    GIT_TAG
    GIT_REMOTE_NAME
    GIT_SUBMODULES
    GIT_SUBMODULES_RECURSE
    GIT_SHALLOW
    GIT_PROGRESS
    GIT_CONFIG
    GIT_REMOTE_UPDATE_STRATEGY
    #
    SVN_REPOSITORY
    SVN_REVISION
    SVN_USERNAME
    SVN_PASSWORD
    SVN_TRUST_CERT
    #
    HG_REPOSITORY
    HG_TAG
    #
    CVS_REPOSITORY
    CVS_MODULE
    CVS_TAG
    #
    # Update step options
    #
    UPDATE_COMMAND
    UPDATE_DISCONNECTED
    #
    # Patch step options
    #
    PATCH_COMMAND
    #
    # Configure step options
    #
    CONFIGURE_COMMAND
    CONFIGURE_ENVIRONMENT_MODIFICATION
    CMAKE_COMMAND
    CMAKE_GENERATOR
    CMAKE_GENERATOR_PLATFORM
    CMAKE_GENERATOR_TOOLSET
    CMAKE_GENERATOR_INSTANCE
    CMAKE_ARGS
    CMAKE_CACHE_ARGS
    CMAKE_CACHE_DEFAULT_ARGS
    SOURCE_SUBDIR
    CONFIGURE_HANDLED_BY_BUILD
    #
    # Build step options
    #
    BUILD_COMMAND
    BUILD_ENVIRONMENT_MODIFICATION
    BUILD_IN_SOURCE
    BUILD_ALWAYS
    BUILD_BYPRODUCTS
    BUILD_JOB_SERVER_AWARE
    #
    # Install step options
    #
    INSTALL_COMMAND
    INSTALL_ENVIRONMENT_MODIFICATION
    INSTALL_BYPRODUCTS
    INSTALL_JOB_SERVER_AWARE
    #
    # Test step options
    #
    TEST_COMMAND
    TEST_ENVIRONMENT_MODIFICATION
    TEST_BEFORE_INSTALL
    TEST_AFTER_INSTALL
    TEST_EXCLUDE_FROM_MAIN
    #
    # Logging options
    #
    LOG_DOWNLOAD
    LOG_UPDATE
    LOG_PATCH
    LOG_CONFIGURE
    LOG_BUILD
    LOG_INSTALL
    LOG_TEST
    LOG_MERGED_STDOUTERR
    LOG_OUTPUT_ON_FAILURE
    #
    # Terminal access options
    #
    USES_TERMINAL_DOWNLOAD
    USES_TERMINAL_UPDATE
    USES_TERMINAL_PATCH
    USES_TERMINAL_CONFIGURE
    USES_TERMINAL_BUILD
    USES_TERMINAL_INSTALL
    USES_TERMINAL_TEST
    #
    # Target options
    #
    DEPENDS
    EXCLUDE_FROM_ALL
    STEP_TARGETS
    INDEPENDENT_STEP_TARGETS
    #
    # Miscellaneous options
    #
    LIST_SEPARATOR
    #
    # Internal options (undocumented)
    #
    EXTERNALPROJECT_INTERNAL_ARGUMENT_SEPARATOR
  )
endmacro()
