IF(CMAKE_SYSTEM_NAME MATCHES "Linux")
  message("-- <FindZVECTOR>")

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
        return (ret == 3856) ? 0 : -1;
    }
   ")

  SET(ARCH_SIMD_TEST_FLAGS " -mvx -mzvector")
  SET(CMAKE_REQUIRED_FLAGS_SAVE ${CMAKE_REQUIRED_FLAGS})
  SET(CMAKE_REQUIRED_FLAGS "${ARCH_SIMD_TEST_FLAGS}")
  # Do compilation check instead of runtime check
  # in case it is compiled on older hardware
  # or crosscompiled
  CHECK_CXX_SOURCE_COMPILES("${VECTORIZATION_CODE}"  COMPILE_OUT_ZVECTOR)
  SET(CMAKE_REQUIRED_FLAGS ${CMAKE_REQUIRED_FLAGS_SAVE})
  if(COMPILE_OUT_ZVECTOR)
    message("-- ZVECTOR flags were set.")
    set(CXX_ZVECTOR_FOUND TRUE)
    SET(CXX_ZVECTOR_FLAGS  "${ARCH_SIMD_TEST_FLAGS}" )
  else()
    message("-- ZVECTOR flags were NOT set.")
  endif()
  message("-- </FindZVECTOR>")

endif()
