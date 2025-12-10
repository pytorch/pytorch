# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.30)
cmake_policy(SET CMP0174 NEW)   # TODO: Remove this when we can update the above to 3.31

function(add_command name test_name)
  set(args "")
  foreach(arg ${ARGN})
    if(arg MATCHES "[^-./:a-zA-Z0-9_]")
      string(APPEND args " [==[${arg}]==]")
    else()
      string(APPEND args " ${arg}")
    endif()
  endforeach()
  string(APPEND script "${name}(${test_name} ${args})\n")
  set(script "${script}" PARENT_SCOPE)
endfunction()

function(generate_testname_guards output open_guard_var close_guard_var)
  set(open_guard "[=[")
  set(close_guard "]=]")
  set(counter 1)
  while("${output}" MATCHES "${close_guard}")
    math(EXPR counter "${counter} + 1")
    string(REPEAT "=" ${counter} equals)
    set(open_guard "[${equals}[")
    set(close_guard "]${equals}]")
  endwhile()
  set(${open_guard_var} "${open_guard}" PARENT_SCOPE)
  set(${close_guard_var} "${close_guard}" PARENT_SCOPE)
endfunction()

function(escape_square_brackets output bracket placeholder placeholder_var output_var)
  if("${output}" MATCHES "\\${bracket}")
    set(placeholder "${placeholder}")
    while("${output}" MATCHES "${placeholder}")
        set(placeholder "${placeholder}_")
    endwhile()
    string(REPLACE "${bracket}" "${placeholder}" output "${output}")
    set(${placeholder_var} "${placeholder}" PARENT_SCOPE)
    set(${output_var} "${output}" PARENT_SCOPE)
  endif()
endfunction()

macro(write_test_to_file)
  # Store the gtest test name before messing with these strings
  set(gtest_name ${current_test_suite}.${current_test_name})

  set(pretty_test_suite ${current_test_suite})
  set(pretty_test_name ${current_test_name})

  # Handle disabled tests
  set(maybe_DISABLED "")
  if(pretty_test_suite MATCHES "^DISABLED_" OR pretty_test_name MATCHES "^DISABLED_")
    set(maybe_DISABLED DISABLED YES)
    string(REGEX REPLACE "^DISABLED_" "" pretty_test_suite "${pretty_test_suite}")
    string(REGEX REPLACE "^DISABLED_" "" pretty_test_name "${pretty_test_name}")
  endif()

  if (NOT current_test_value_param STREQUAL "" AND NOT arg_NO_PRETTY_VALUES)
    # Remove value param name, if any, from test name
    string(REGEX REPLACE "^(.+)/.+$" "\\1" pretty_test_name "${pretty_test_name}")
    set(pretty_test_name "${pretty_test_name}/${current_test_value_param}")
  endif()

  if(NOT current_test_type_param STREQUAL "")
    # Parse type param name from suite name
    if(pretty_test_suite MATCHES "^(.+)/(.+)$")
      set(pretty_test_suite "${CMAKE_MATCH_1}")
      set(current_type_param_name "${CMAKE_MATCH_2}")
    else()
      set(current_type_param_name "")
    endif()
    if (NOT arg_NO_PRETTY_TYPES)
      string(APPEND pretty_test_name  "<${current_test_type_param}>")
    elseif(NOT current_type_param_name STREQUAL "")
        string(APPEND pretty_test_name "<${current_type_param_name}>")
    endif()
  endif()

  set(test_name_template "@prefix@@pretty_test_suite@.@pretty_test_name@@suffix@")
  string(CONFIGURE "${test_name_template}" testname)

  if(NOT "${arg_TEST_XML_OUTPUT_DIR}" STREQUAL "")
    set(TEST_XML_OUTPUT_PARAM "--gtest_output=xml:${arg_TEST_XML_OUTPUT_DIR}/${prefix}${gtest_name}${suffix}.xml")
  else()
    set(TEST_XML_OUTPUT_PARAM "")
  endif()

  # unescape []
  if(open_sb)
    string(REPLACE "${open_sb}" "[" testname "${testname}")
  endif()
  if(close_sb)
    string(REPLACE "${close_sb}" "]" testname "${testname}")
  endif()
  set(guarded_testname "${open_guard}${testname}${close_guard}")
  # Add to script. Do not use add_command() here because it messes up the
  # handling of empty values when forwarding arguments, and we need to
  # preserve those carefully for arg_TEST_EXECUTOR and arg_EXTRA_ARGS.
  string(APPEND script "add_test(${guarded_testname} ${launcherArgs}")
  foreach(arg IN ITEMS
    "${arg_TEST_EXECUTABLE}"
    "--gtest_filter=${gtest_name}"
    "--gtest_also_run_disabled_tests"
    ${TEST_XML_OUTPUT_PARAM}
  )

    if(arg MATCHES "[^-./:a-zA-Z0-9_]")
      string(APPEND script " [==[${arg}]==]")
    else()
      string(APPEND script " ${arg}")
    endif()
  endforeach()

  if(arg_TEST_EXTRA_ARGS)
    list(JOIN arg_TEST_EXTRA_ARGS "]==] [==[" extra_args)
    string(APPEND script " [==[${extra_args}]==]")
  endif()
  string(APPEND script ")\n")

  set(maybe_LOCATION "")
  if(NOT current_test_file STREQUAL "" AND NOT current_test_line STREQUAL "")
    set(maybe_LOCATION DEF_SOURCE_LINE "${current_test_file}:${current_test_line}")
  endif()

  add_command(set_tests_properties
    "${guarded_testname}"
    PROPERTIES
      ${maybe_DISABLED}
      ${maybe_LOCATION}
      WORKING_DIRECTORY "${arg_TEST_WORKING_DIR}"
      SKIP_REGULAR_EXPRESSION "\\[  SKIPPED \\]"
      ${arg_TEST_PROPERTIES}
  )

  # possibly unbalanced square brackets render lists invalid so skip such
  # tests in ${arg_TEST_LIST}
  if(NOT "${testname}" MATCHES [=[(\[|\])]=])
    # escape ;
    string(REPLACE [[;]] [[\\;]] testname "${testname}")
    list(APPEND tests_buffer "${testname}")
    list(LENGTH tests_buffer tests_buffer_length)
    if(tests_buffer_length GREATER "250")
      # Chunk updates to the final "tests" variable, keeping the
      # "tests_buffer" variable that we append each test to relatively
      # small. This mitigates worsening performance impacts for the
      # corner case of having many thousands of tests.
      list(APPEND tests "${tests_buffer}")
      set(tests_buffer "")
    endif()
  endif()

  # If we've built up a sizable script so far, write it out as a chunk now
  # so we don't accumulate a massive string to write at the end
  string(LENGTH "${script}" script_len)
  if(${script_len} GREATER "50000")
    file(APPEND "${arg_CTEST_FILE}" "${script}")
    set(script "")
  endif()
endmacro()

macro(parse_tests_from_output)
  generate_testname_guards("${output}" open_guard close_guard)
  escape_square_brackets("${output}" "[" "__osb" open_sb output)
  escape_square_brackets("${output}" "]" "__csb" close_sb output)

  # Preserve semicolon in test-parameters
  string(REPLACE [[;]] [[\;]] output "${output}")
  string(REPLACE "\n" ";" output "${output}")

  # Command line output doesn't contain information about the file and line number of the tests
  set(current_test_file "")
  set(current_test_line "")

  # Parse output
  foreach(line ${output})
    # Skip header
    if(line MATCHES "gtest_main\\.cc")
      continue()
    endif()

    if(line STREQUAL "")
      continue()
    endif()

    # Do we have a module name or a test name?
    if(NOT line MATCHES "^  ")
      set(current_test_type_param "")

      # Module; remove trailing '.' to get just the name...
      string(REGEX REPLACE "\\.( *#.*)?$" "" current_test_suite "${line}")
      if(line MATCHES "# *TypeParam = (.*)$")
        set(current_test_type_param "${CMAKE_MATCH_1}")
      endif()
    else()
      string(STRIP "${line}" test)
      string(REGEX REPLACE " ( *#.*)?$" "" current_test_name "${test}")

      set(current_test_value_param "")
      if(line MATCHES "# *GetParam\\(\\) = (.*)$")
        set(current_test_value_param "${CMAKE_MATCH_1}")
      endif()

      write_test_to_file()
    endif()
  endforeach()
endmacro()

macro(get_json_member_with_default json_variable member_name out_variable)
  string(JSON ${out_variable}
    ERROR_VARIABLE error_param
    GET "${${json_variable}}" "${member_name}"
  )
  if(error_param)
    # Member not present
    set(${out_variable} "")
  endif()
endmacro()

macro(parse_tests_from_json json_file)
  if(NOT EXISTS "${json_file}")
    message(FATAL_ERROR "Missing expected JSON file with test list: ${json_file}")
  endif()

  file(READ "${json_file}" test_json)
  string(JSON test_suites_json GET "${test_json}" "testsuites")

  # Return if there are no testsuites
  string(JSON len_test_suites LENGTH "${test_suites_json}")
  if(len_test_suites GREATER 0)
    set(open_sb)
    set(close_sb)

    math(EXPR upper_limit_test_suite_range "${len_test_suites} - 1")

    foreach(index_test_suite RANGE ${upper_limit_test_suite_range})
      string(JSON test_suite_json GET "${test_suites_json}" ${index_test_suite})

      # "suite" is expected to be set in write_test_to_file(). When parsing the
      # plain text output, "suite" is expected to be the original suite name
      # before accounting for pretty names. This may be used to construct the
      # name of XML output results files.
      string(JSON current_test_suite GET "${test_suite_json}" "name")
      string(JSON tests_json GET "${test_suite_json}" "testsuite")

      # Skip test suites without tests
      string(JSON len_tests LENGTH "${tests_json}")
      if(len_tests LESS_EQUAL 0)
        continue()
      endif()

      math(EXPR upper_limit_test_range "${len_tests} - 1")
      foreach(index_test RANGE ${upper_limit_test_range})
        string(JSON test_json GET "${tests_json}" ${index_test})

        string(JSON len_test_parameters LENGTH "${test_json}")
        if(len_test_parameters LESS_EQUAL 0)
          continue()
        endif()

        get_json_member_with_default(test_json "name" current_test_name)
        get_json_member_with_default(test_json "file" current_test_file)
        get_json_member_with_default(test_json "line" current_test_line)
        get_json_member_with_default(test_json "value_param" current_test_value_param)
        get_json_member_with_default(test_json "type_param" current_test_type_param)

        generate_testname_guards(
          "${current_test_suite}${current_test_name}${current_test_value_param}${current_test_type_param}"
          open_guard close_guard
        )
        write_test_to_file()
      endforeach()
    endforeach()
  endif()
endmacro()

function(gtest_discover_tests_impl)

  set(options "")
  set(oneValueArgs
    NO_PRETTY_TYPES   # These two take a value, unlike gtest_discover_tests()
    NO_PRETTY_VALUES  #
    TEST_TARGET
    TEST_EXECUTABLE
    TEST_WORKING_DIR
    TEST_PREFIX
    TEST_SUFFIX
    TEST_LIST
    CTEST_FILE
    TEST_DISCOVERY_TIMEOUT
    TEST_XML_OUTPUT_DIR
    # The following are all multi-value arguments in gtest_discover_tests(),
    # but they are each given to us as a single argument. We parse them that
    # way to avoid problems with preserving empty list values and escaping.
    TEST_FILTER
    TEST_EXTRA_ARGS
    TEST_DISCOVERY_EXTRA_ARGS
    TEST_PROPERTIES
    TEST_EXECUTOR
  )
  set(multiValueArgs "")
  cmake_parse_arguments(PARSE_ARGV 0 arg
    "${options}" "${oneValueArgs}" "${multiValueArgs}"
  )

  set(prefix "${arg_TEST_PREFIX}")
  set(suffix "${arg_TEST_SUFFIX}")
  set(script)
  set(tests)
  set(tests_buffer "")

  # If a file at ${arg_CTEST_FILE} already exists, we overwrite it.
  file(REMOVE "${arg_CTEST_FILE}")

  set(filter)
  if(arg_TEST_FILTER)
    set(filter "--gtest_filter=${arg_TEST_FILTER}")
  endif()

  # CMP0178 has already been handled in gtest_discover_tests(), so we only need
  # to implement NEW behavior here. This means preserving empty arguments for
  # TEST_EXECUTOR. For OLD or WARN, gtest_discover_tests() already removed any
  # empty arguments.
  set(launcherArgs "")
  if(NOT "${arg_TEST_EXECUTOR}" STREQUAL "")
    list(JOIN arg_TEST_EXECUTOR "]==] [==[" launcherArgs)
    set(launcherArgs "[==[${launcherArgs}]==]")
  endif()

  # Run test executable to get list of available tests
  if(NOT EXISTS "${arg_TEST_EXECUTABLE}")
    message(FATAL_ERROR
      "Specified test executable does not exist.\n"
      "  Path: '${arg_TEST_EXECUTABLE}'"
    )
  endif()

  set(discovery_extra_args "")
  if(NOT "${arg_TEST_DISCOVERY_EXTRA_ARGS}" STREQUAL "")
    list(JOIN arg_TEST_DISCOVERY_EXTRA_ARGS "]==] [==[" discovery_extra_args)
    set(discovery_extra_args "[==[${discovery_extra_args}]==]")
  endif()

  # Avoid a potential race condition for the POST_BUILD case when multiple
  # calls are made to gtest_discover_tests() for different targets but the same
  # working directory. For PRE_TEST, we're always executing serially during the
  # ctest setup phase, so there is no race condition there, but POST_BUILD can
  # lead to this code path being run in parallel. Use a hash to avoid potential
  # problems with very long target names.
  string(SHA256 target_hash "${arg_TEST_TARGET}")
  string(SUBSTRING "${target_hash}" 0 10 target_hash)
  set(json_file
    "${arg_TEST_WORKING_DIR}/cmake_test_discovery_${target_hash}.json"
  )

  # Remove json file to make sure we don't pick up an outdated one
  file(REMOVE "${json_file}")

  cmake_language(EVAL CODE
    "execute_process(
      COMMAND ${launcherArgs} [==[${arg_TEST_EXECUTABLE}]==]
        --gtest_list_tests
        [==[--gtest_output=json:${json_file}]==]
        ${filter}
        ${discovery_extra_args}
      WORKING_DIRECTORY [==[${arg_TEST_WORKING_DIR}]==]
      TIMEOUT ${arg_TEST_DISCOVERY_TIMEOUT}
      OUTPUT_VARIABLE output
      RESULT_VARIABLE result
    )"
  )

  if(NOT ${result} EQUAL 0)
    string(REPLACE "\n" "\n    " output "${output}")
    if(arg_TEST_EXECUTOR)
      set(path "${arg_TEST_EXECUTOR} ${arg_TEST_EXECUTABLE}")
    else()
      set(path "${arg_TEST_EXECUTABLE}")
    endif()
    message(FATAL_ERROR
      "Error running test executable.\n"
      "  Path: '${path}'\n"
      "  Working directory: '${arg_TEST_WORKING_DIR}'\n"
      "  Result: ${result}\n"
      "  Output:\n"
      "    ${output}\n"
    )
  endif()

  if(EXISTS "${json_file}")
    parse_tests_from_json("${json_file}")
  else()
    # gtest < 1.8.1, and all gtest compiled with GTEST_HAS_FILE_SYSTEM=0, don't
    # recognize the --gtest_output=json option, and issue a warning or error on
    # stdout about it being unrecognized, but still return an exit code 0 for
    # success. All versions report the test list on stdout whether
    # --gtest_output=json is recognized or not.

    # NOTE: Because we are calling a macro, we don't want to pass "output" as
    # an argument because it messes up the contents passed through due to the
    # different escaping, etc. that gets applied. We rely on it picking up the
    # "output" variable we have already set here.
    parse_tests_from_output()
  endif()

  if(NOT tests_buffer STREQUAL "")
    list(APPEND tests "${tests_buffer}")
  endif()

  # Create a list of all discovered tests, which users may use to e.g. set
  # properties on the tests
  add_command(set "" ${arg_TEST_LIST} "${tests}")

  # Write remaining content to the CTest script
  file(APPEND "${arg_CTEST_FILE}" "${script}")
endfunction()

if(CMAKE_SCRIPT_MODE_FILE)
  gtest_discover_tests_impl(
    NO_PRETTY_TYPES ${NO_PRETTY_TYPES}
    NO_PRETTY_VALUES ${NO_PRETTY_VALUES}
    TEST_TARGET ${TEST_TARGET}
    TEST_EXECUTABLE ${TEST_EXECUTABLE}
    TEST_EXECUTOR "${TEST_EXECUTOR}"
    TEST_WORKING_DIR ${TEST_WORKING_DIR}
    TEST_PREFIX ${TEST_PREFIX}
    TEST_SUFFIX ${TEST_SUFFIX}
    TEST_FILTER ${TEST_FILTER}
    TEST_LIST ${TEST_LIST}
    CTEST_FILE ${CTEST_FILE}
    TEST_DISCOVERY_TIMEOUT ${TEST_DISCOVERY_TIMEOUT}
    TEST_XML_OUTPUT_DIR ${TEST_XML_OUTPUT_DIR}
    TEST_EXTRA_ARGS "${TEST_EXTRA_ARGS}"
    TEST_DISCOVERY_EXTRA_ARGS "${TEST_DISCOVERY_EXTRA_ARGS}"
    TEST_PROPERTIES "${TEST_PROPERTIES}"
  )
endif()
