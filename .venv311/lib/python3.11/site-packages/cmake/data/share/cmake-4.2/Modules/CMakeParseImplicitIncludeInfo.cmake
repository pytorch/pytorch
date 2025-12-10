# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

# This is used internally by CMake and should not be included by user code.

# helper function that parses implicit include dirs from a single line
# for compilers that report them that way.  on success we return the
# list of dirs in id_var and set state_var to the 'done' state.
function(cmake_parse_implicit_include_line line lang id_var log_var state_var)
  # clear variables we append to (avoids possible pollution from parent scopes)
  unset(rv)
  set(log "")

  # Cray compiler (from cray wrapper, via PrgEnv-cray)
  if(CMAKE_${lang}_COMPILER_ID STREQUAL "Cray" AND
     line MATCHES "^/" AND line MATCHES "/ccfe |/ftnfe " AND
     line MATCHES " -isystem| -I")
    string(REGEX MATCHALL " (-I ?|-isystem )(\"[^\"]+\"|[^ \"]+)" incs "${line}")
    foreach(inc IN LISTS incs)
      string(REGEX REPLACE " (-I ?|-isystem )(\"[^\"]+\"|[^ \"]+)" "\\2" idir "${inc}")
      list(APPEND rv "${idir}")
    endforeach()
    if(rv)
      string(APPEND log "  got implicit includes via cray ccfe parser!\n")
    else()
      string(APPEND log "  warning: cray ccfe parse failed!\n")
    endif()
  endif()

  # PGI compiler
  if(CMAKE_${lang}_COMPILER_ID STREQUAL "PGI")
    # pgc++ verbose output differs
    if((lang STREQUAL "C" OR lang STREQUAL "Fortran") AND
        line MATCHES "^/" AND
        line MATCHES "/pgc |/pgf901 |/pgftnc " AND
        line MATCHES " -cmdline ")
      # cmdline has unparsed cmdline, remove it
      string(REGEX REPLACE "-cmdline .*" "" line "${line}")
      if("${line}" MATCHES " -nostdinc ")
        set(rv "")    # defined, but empty
      else()
        string(REGEX MATCHALL " -stdinc ([^ ]*)" incs "${line}")
        foreach(inc IN LISTS incs)
          string(REGEX REPLACE " -stdinc ([^ ]*)" "\\1" idir "${inc}")
          string(REPLACE ":" ";" idir "${idir}")
          list(APPEND rv ${idir})
        endforeach()
      endif()
      if(DEFINED rv)
        string(APPEND log "  got implicit includes via PGI C/F parser!\n")
      else()
        string(APPEND log "  warning: PGI C/F parse failed!\n")
      endif()
    elseif(lang STREQUAL "CXX" AND line MATCHES "^/" AND
           line MATCHES "/pggpp1 " AND line MATCHES " -I")
      # oddly, -Mnostdinc does not get rid of system -I's, at least in
      # PGI 18.10.1 ...
      string(REGEX MATCHALL " (-I ?)([^ ]*)" incs "${line}")
      foreach(inc IN LISTS incs)
        string(REGEX REPLACE " (-I ?)([^ ]*)" "\\2" idir "${inc}")
        if(NOT idir STREQUAL "-")   # filter out "-I-"
          list(APPEND rv "${idir}")
        endif()
      endforeach()
      if(DEFINED rv)
        string(APPEND log "  got implicit includes via PGI CXX parser!\n")
      else()
        string(APPEND log "  warning: PGI CXX parse failed!\n")
      endif()
    endif()
  endif()

  # SunPro compiler
  if(CMAKE_${lang}_COMPILER_ID STREQUAL "SunPro" AND
     (line MATCHES "-D__SUNPRO_C" OR line MATCHES "-D__SUNPRO_F"))
    string(REGEX MATCHALL " (-I ?)([^ ]*)" incs "${line}")
    foreach(inc IN LISTS incs)
      string(REGEX REPLACE " (-I ?)([^ ]*)" "\\2" idir "${inc}")
      if(NOT "${idir}" STREQUAL "-xbuiltin")
        list(APPEND rv "${idir}")
      endif()
    endforeach()
    if(rv)
      if (lang STREQUAL "C" OR lang STREQUAL "CXX")
        # /usr/include appears to be hardwired in
        list(APPEND rv "/usr/include")
      endif()
      string(APPEND log "  got implicit includes via sunpro parser!\n")
    else()
      string(APPEND log "  warning: sunpro parse failed!\n")
    endif()
  endif()

  # XL compiler
  if((CMAKE_${lang}_COMPILER_ID STREQUAL "XL"
      OR CMAKE_${lang}_COMPILER_ID STREQUAL "XLClang")
     AND line MATCHES "^/"
     AND ( (lang STREQUAL "Fortran" AND
            line MATCHES "/xl[fF]entry " AND
            line MATCHES "OSVAR\\([^ ]+\\)")
           OR
            ( (lang STREQUAL "C" OR lang STREQUAL "CXX") AND
            line MATCHES "/xl[cC]2?entry " AND
            line MATCHES " -qosvar=")
         )  )
    # -qnostdinc cancels other stdinc flags, even if present
    string(FIND "${line}" " -qnostdinc" nostd)
    if(NOT nostd EQUAL -1)
      set(rv "")    # defined but empty
      string(APPEND log "  got implicit includes via XL parser (nostdinc)\n")
    else()
      if(lang STREQUAL "CXX")
        string(REGEX MATCHALL " -qcpp_stdinc=([^ ]*)" std "${line}")
        string(REGEX MATCHALL " -qgcc_cpp_stdinc=([^ ]*)" gcc_std "${line}")
      else()
        string(REGEX MATCHALL " -qc_stdinc=([^ ]*)" std "${line}")
        string(REGEX MATCHALL " -qgcc_c_stdinc=([^ ]*)" gcc_std "${line}")
      endif()
      set(xlstd ${std} ${gcc_std})
      foreach(inc IN LISTS xlstd)
        string(REGEX REPLACE " -q(cpp|gcc_cpp|c|gcc_c)_stdinc=([^ ]*)" "\\2"
               ipath "${inc}")
        string(REPLACE ":" ";" ipath "${ipath}")
        list(APPEND rv ${ipath})
      endforeach()
    endif()
    # user can add -I flags via CMAKE_{C,CXX}_FLAGS, look for that too
    string(REGEX MATCHALL " (-I ?)([^ ]*)" incs "${line}")
    unset(urv)
    foreach(inc IN LISTS incs)
      string(REGEX REPLACE " (-I ?)([^ ]*)" "\\2" idir "${inc}")
      list(APPEND urv "${idir}")
    endforeach()
    if(urv)
      if ("${rv}" STREQUAL "")
        set(rv ${urv})
      else()
        list(APPEND rv ${urv})
      endif()
    endif()

    if(DEFINED rv)
      string(APPEND log "  got implicit includes via XL parser!\n")
    else()
      string(APPEND log "  warning: XL parse failed!\n")
    endif()
  endif()

  # Fujitsu compiler
  if(CMAKE_${lang}_COMPILER_ID STREQUAL "Fujitsu" AND
     line MATCHES "/ccpcom")
    string(REGEX MATCHALL " (-I *|--sys_include=|--preinclude +)(\"[^\"]+\"|[^ \"]+)" incs "${line}")
    foreach(inc IN LISTS incs)
      string(REGEX REPLACE " (-I *|--sys_include=|--preinclude +)(\"[^\"]+\"|[^ \"]+)" "\\2" idir "${inc}")
      list(APPEND rv "${idir}")
    endforeach()
    if(rv)
      string(APPEND log "  got implicit includes via fujitsu ccpcom parser!\n")
    else()
      string(APPEND log "  warning: fujitsu ccpcom parse failed!\n")
    endif()
  endif()

  if(log)
    set(${log_var} "${log}" PARENT_SCOPE)
  else()
    unset(${log_var} PARENT_SCOPE)
  endif()
  if(DEFINED rv)
    set(${id_var} "${rv}" PARENT_SCOPE)
    set(${state_var} "done" PARENT_SCOPE)
  endif()
endfunction()

# top-level function to parse implicit include directory information
# from verbose compiler output. sets state_var in parent to 'done' on success.
function(cmake_parse_implicit_include_info text lang dir_var log_var state_var)
  set(state start)    # values: start, loading, done

  # clear variables we append to (avoids possible pollution from parent scopes)
  set(implicit_dirs_tmp)
  set(log "")

  # go through each line of output...
  string(REGEX REPLACE "\r*\n" ";" output_lines "${text}")
  foreach(line IN LISTS output_lines)
    if(state STREQUAL start)
      string(FIND "${line}" "#include \"...\" search starts here:" rv)
      if(rv GREATER -1)
        set(state loading)
        set(preload 1)      # looking for include <...> now
        string(APPEND log "  found start of include info\n")
      else()
        cmake_parse_implicit_include_line("${line}" "${lang}" implicit_dirs_tmp
                                          linelog state)
        if(linelog)
          string(APPEND log ${linelog})
        endif()
        if(state STREQUAL done)
          break()
        endif()
      endif()
    elseif(state STREQUAL loading)
      string(FIND "${line}" "End of search list." rv)
      if(rv GREATER -1)
        set(state done)
        string(APPEND log "  end of search list found\n")
        break()
      endif()
      if(preload)
        string(FIND "${line}" "#include <...> search starts here:" rv)
        if(rv GREATER -1)
          set(preload 0)
          string(APPEND log "  found start of implicit include info\n")
        endif()
        continue()
      endif()
      if("${line}" MATCHES "^ ")
        string(SUBSTRING "${line}" 1 -1 line)  # remove leading space
      endif()
      if ("${line}" MATCHES " \\(framework directory\\)$")
        continue() # frameworks are handled elsewhere, ignore them here
      endif()
      string(REPLACE "\\" "/" path "${line}")
      list(APPEND implicit_dirs_tmp "${path}")
      string(APPEND log "    add: [${path}]\n")
    endif()
  endforeach()

  set(implicit_dirs "")
  foreach(d IN LISTS implicit_dirs_tmp)
    if(IS_ABSOLUTE "${d}")
      get_filename_component(dir "${d}" ABSOLUTE)
      list(APPEND implicit_dirs "${dir}")
      string(APPEND log "  collapse include dir [${d}] ==> [${dir}]\n")
    elseif("${d}" MATCHES [[^\.\.[\/]\.\.[\/]\.\.[\/](.*)$]])
      # This relative path is deep enough to get out of the
      #     CMakeFiles/CMakeScratch/<unique>
      # directory where the ABI check is done.  Assume that the compiler has
      # computed this path adaptively based on the current working directory
      # such that the effective result is absolute.
      get_filename_component(dir "${CMAKE_BINARY_DIR}/${CMAKE_MATCH_1}" ABSOLUTE)
      list(APPEND implicit_dirs "${dir}")
      string(APPEND log "  collapse relative include dir [${d}] ==> [${dir}]\n")
    else()
      string(APPEND log "  skipping relative include dir [${d}]\n")
    endif()
  endforeach()
  list(REMOVE_DUPLICATES implicit_dirs)

  # Log results.
  if(state STREQUAL done)
    string(APPEND log "  implicit include dirs: [${implicit_dirs}]\n")
  else()
    string(APPEND log "  warn: unable to parse implicit include dirs!\n")
  endif()

  # Return results.
  set(${dir_var} "${implicit_dirs}" PARENT_SCOPE)
  set(${log_var} "${log}" PARENT_SCOPE)
  set(${state_var} "${state}" PARENT_SCOPE)

endfunction()
