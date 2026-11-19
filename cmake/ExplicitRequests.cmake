##############################################################################
# Explicit-request tracking for build options. this file is included by the
# top-level CMakeLists.txt only and is not installed: find_package(Torch)
# consumers never declare pytorch options, so only the
# pytorch_guard_explicit_request() they can reach from the installed
# cmake/public/utils.cmake stays public.
##############################################################################
# Record whether <option> was set by the user and not just defaulted.
#
# call this right above the option's own option() / cmake_dependent_option()
# line, passing the doc string that declaration uses. the cache entry found
# there was set by the user (-D or an initial cache script with its own
# wording), forwarded from the env by cmake/EnvVarForwarding.cmake, or
# written by an earlier configure's declaration or fallback, and only the
# first kind carries something other than the declaration's doc string:
# option() and
# cmake_dependent_option() stamp their doc string over an entry they adopt,
# caffe2_update_option() keeps it. only checking whether the entry exists
# gets this wrong: a configure that dies before its fallback runs leaves the
# declaration default behind and the next run reads it as a user request.
#
# three signals decide, in order:
#   1. the help string is not the declaration's own (-D entry, env
#      forwarding, a -C script with its own wording): whatever wrote the
#      entry, the value is the user's intent.
#   2. the help string matches but the cached value moved since the
#      last write the build itself made (tracked in the
#      _<option>_LAST_SEEN INTERNAL entry): option() never rewrites the
#      value of an entry it adopts, and every caffe2_update_option()
#      write restamps it with what it wrote, so a value that differs
#      from the stamp is a ccmake / cmake-gui / hand edit of
#      CMakeCache.txt, which is gh-114388's exact symptom. editing the
#      value back to OFF this way withdraws the request on the run that
#      sees it, which the early refusals in CMakeLists.txt read. the
#      comparison needs an existing stamp: a build dir from before the
#      stamping (or a -C script that quoted the doc string into a fresh
#      dir) has none, and reading the missing stamp as a value would
#      record a request nobody made -- those entries are left to the
#      declaration's own judgment, so a -C script that reuses the doc
#      string is not recorded.
#   3. otherwise the value is stable since the declaration last ran, or
#      there is no stamp yet to compare against: a declaration default or
#      an already-recorded request, and the marker is left alone.
#
# the decision is kept in an INTERNAL marker that survives the re-stamp:
# the declaration stamps its own doc string over an adopted entry during
# the same run, and the cache is saved even when a later guard fails the
# configure, so the next run in the same dir would read the stamped entry
# as a declaration default and forget the request.
#
# remaining limits, on purpose: a cache entry whose help string and value
# both impersonate the declaration's default is indistinguishable from
# one and always was; env forwarding that rewrites a value the cache
# already holds is a no-op and leaves no new signal, so a build dir whose
# earlier configure died before the CUDA fallback falls back silently
# where a fresh dir would fail; a forced dependent-option value lives in
# a normal variable that the cache never sees, so the request state reads
# the cache; and caffe2_update_option() writes restamp the value they
# write (an ON re-affirming the declaration, FBGEMM and NUMPY, or an OFF
# flipped to ON to match an enabled backend, SYSTEM_NCCL), so they cannot
# invent or erase a request.
function(pytorch_record_explicit_request option docstring)
  if(DEFINED CACHE{${option}})
    get_property(_help CACHE ${option} PROPERTY HELPSTRING)
    set(_last_seen "$CACHE{_${option}_LAST_SEEN}")
    set(_request "")
    if(NOT _help STREQUAL "${docstring}")
      set(_request "${${option}}")
    elseif(DEFINED CACHE{_${option}_LAST_SEEN}
           AND NOT "${${option}}" STREQUAL "${_last_seen}")
      set(_request "${${option}}")
    endif()
    if(NOT _request STREQUAL "")
      if(_request)
        set(_${option}_EXPLICITLY_SET ON
            CACHE INTERNAL "user-sourced request for ${option}" FORCE)
      else()
        set(_${option}_EXPLICITLY_SET OFF
            CACHE INTERNAL "user-sourced request for ${option}" FORCE)
      endif()
    endif()
  else()
    # removing the entry (cmake -U, or "Remove Entry" in cmake-gui) is the
    # standard way back to the default, so the request goes with it
    unset(_${option}_EXPLICITLY_SET CACHE)
    unset(_${option}_LAST_SEEN CACHE)
  endif()
  set(_${option}_EXPLICITLY_SET "${_${option}_EXPLICITLY_SET}" PARENT_SCOPE)
endfunction()


##############################################################################
# option() / cmake_dependent_option() wrappers that record an explicit user
# request first. the record call needs the declaration's doc string to tell
# a user entry apart from an earlier declaration default, so the two have
# to stay in one place: use these instead of calling
# pytorch_record_explicit_request() by hand above a plain option() or
# cmake_dependent_option() line. the _LAST_SEEN stamp after the declaration
# is what lets the next run tell a cache edit apart from the declaration's
# own write: it records the value the option settled on while the
# declaration was running, before any fallback touches it.
macro(pytorch_option option docstring default_value)
  pytorch_record_explicit_request(${option} "${docstring}")
  option(${option} "${docstring}" ${default_value})
  set(_${option}_LAST_SEEN "$CACHE{${option}}" CACHE INTERNAL
      "value ${option} settled on" FORCE)
endmacro()


macro(pytorch_dependent_option option docstring default_value depends force)
  pytorch_record_explicit_request(${option} "${docstring}")
  cmake_dependent_option(${option} "${docstring}" ${default_value}
                         "${depends}" ${force})
  # stamp what the cache holds, not ${option}: the dependent option's
  # force value lands in a normal variable while the cache entry keeps
  # its old value, and a stamp that disagrees with the cache would read
  # as a user edit on the next configure
  set(_${option}_LAST_SEEN "$CACHE{${option}}" CACHE INTERNAL
      "value ${option} settled on" FORCE)
endmacro()


##############################################################################
# the declaration-site counterpart of pytorch_guard_explicit_request():
# fail when <option> was recorded as explicitly requested, with the shared
# refusal wording. no-op otherwise. the recorded request, not the current
# value, decides: the dependent option may have already forced an
# explicitly requested ON to OFF on a platform it does not support, and
# the refusal still has to fire
function(pytorch_refuse_explicit_request option feature reason)
  if(_${option}_EXPLICITLY_SET)
    message(FATAL_ERROR
      "PyTorch: ${feature} was explicitly requested (${option}=1) but "
      "${reason}. "
      "If you want to build without ${feature}, please set ${option}=0.")
  endif()
endfunction()
