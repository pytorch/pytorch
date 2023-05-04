
IF(CMAKE_SYSTEM_NAME MATCHES "Linux")
  message("-- <FindZVECTOR>")
  set(Z_ARCH_LIST "")
  #firstly, tries to add the arch of the platform
  EXEC_PROGRAM(LD_SHOW_AUXV=1 ARGS "/bin/true" OUTPUT_VARIABLE bintrue)
  if(bintrue MATCHES "AT_PLATFORM:[ \\t\\n\\r]*([a-zA-Z0-9_]+)[ \\t\\n\\r]*")
    if(CMAKE_MATCH_COUNT GREATER 0)
      string(TOLOWER ${CMAKE_MATCH_1} platform)
      if(${platform} MATCHES "^z(14|15|16)")
        message("-- Z ARCH Platform: ${platform}")
        list( APPEND Z_ARCH_LIST  "${platform}" )
      endif()
    endif()
  endif()
  #adds other archs in descending order. as its cached nothing will be checked  twice
  list( APPEND Z_ARCH_LIST  "z16" )
  list( APPEND Z_ARCH_LIST  "z15" )
  list( APPEND Z_ARCH_LIST  "z14" )

  SET(VECTORIZATION_CODE  "
    #include <vecintrin.h>
    using vuint32  =  __attribute__ ((vector_size (16)))  unsigned  int;
    using vfloat32 =  __attribute__ ((vector_size (16)))  float;
    vfloat32 vsel_ext(vuint32 o, vfloat32 x, vfloat32 y)
    {
        return vec_sel(y, x, o);
    }
    int main(){
        vfloat32 h1 ={3.f, 4.f, 5.f, 6.f};
        vfloat32 h2 = {9.f, 8.f, 11.f, 12.f};
        vuint32  selector= {0xFFFFFFFF, 0, 0xFFFFFFFF, 0xFFFFFFFF};
        vfloat32 hf = vsel_ext(selector, h1,h2);
        int ret = (int)(hf[0]*1000+hf[1]*100+hf[2]*10+hf[3]);
        return ret==3856;
    }
   ")

  foreach(Z_ARCH  ${Z_ARCH_LIST})
    SET(ARCH_SIMD_TEST_FLAGS_${Z_ARCH} " -mvx -mzvector -march=${Z_ARCH} -mtune=${Z_ARCH}")
    message("-- check ${Z_ARCH}")
    SET(CMAKE_REQUIRED_FLAGS_SAVE ${CMAKE_REQUIRED_FLAGS})
    SET(CMAKE_REQUIRED_FLAGS "${ARCH_SIMD_TEST_FLAGS_${Z_ARCH}}")
    set(VECTORIZATION_CODE_${Z_ARCH} "${VECTORIZATION_CODE}")
    CHECK_CXX_SOURCE_COMPILES("${VECTORIZATION_CODE_${Z_ARCH}}"  COMPILE_OUT_${Z_ARCH})
    SET(CMAKE_REQUIRED_FLAGS ${CMAKE_REQUIRED_FLAGS_SAVE})
    if(COMPILE_OUT_${Z_ARCH})
      message("-- ${Z_ARCH} SIMD flags were set.")
      set(CXX_ZVECTOR_FOUND TRUE)
      SET(CXX_ZVECTOR_FLAGS  "${ARCH_SIMD_TEST_FLAGS_${Z_ARCH}}" )
      break()
    endif()
  endforeach()
  message("-- </FindZVECTOR>")

endif()
