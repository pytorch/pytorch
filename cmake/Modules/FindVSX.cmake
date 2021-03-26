
IF(CMAKE_SYSTEM_NAME MATCHES "Linux")
  message("-- <FindVSX>")
  EXEC_PROGRAM(LD_SHOW_AUXV=1 ARGS "/bin/true" OUTPUT_VARIABLE bintrue)
  if(bintrue MATCHES "AT_PLATFORM:[ \\t\\n\\r]*([a-zA-Z0-9_]+)[ \\t\\n\\r]*")
    if(CMAKE_MATCH_COUNT GREATER 0)
      string(TOLOWER ${CMAKE_MATCH_1} platform)
      if(${platform} MATCHES "^power")
        message("-- POWER Platform: ${platform}")
        SET(POWER_COMP TRUE CACHE BOOL "power ")
        SET(CXX_VSX_FLAGS  "${CXX_VSX_FLAGS} -mcpu=${platform} -mtune=${platform}" )
      endif()
    endif()
  endif()
  SET(VSX_CODE " #include <altivec.h>
      int main() {
      float __attribute__((aligned(16))) vptr_y[8]   = { 1.0f,2.f,3.f,4.f,4.f,3.f,2.f,1.f };
      __vector float v_result = vec_add(vec_vsx_ld(0, vptr_y), vec_vsx_ld(16, vptr_y));
      return 0;
      }")
  #check_cxx_compiler_flag(-mvsx vsx_flag)
  SET(CMAKE_REQUIRED_FLAGS_SAVE ${CMAKE_REQUIRED_FLAGS})
  SET(CMAKE_REQUIRED_FLAGS "-mvsx")
  CHECK_C_SOURCE_COMPILES("${VSX_CODE}"  C_VSX_FOUND)
  CHECK_CXX_SOURCE_COMPILES("${VSX_CODE}"  CXX_VSX_FOUND)
  SET(CMAKE_REQUIRED_FLAGS ${CMAKE_REQUIRED_FLAGS_SAVE})
  if(CXX_VSX_FOUND)
    message("-- VSX flag was set.")
    SET(CXX_VSX_FLAGS  "${CXX_VSX_FLAGS} -mvsx" )
  elseif(POWER_COMP)
    message(WARNING "-- VSX flag was not set.")
  endif()
  message("-- </FindVSX>")
endif()

