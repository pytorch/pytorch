IF(CMAKE_SYSTEM_NAME MATCHES "Linux")
  message("-- <FindRVV>")
  SET(RVV_CODE "
    #include <riscv_vector.h>
    #if defined(__riscv_v_intrinsic) &&  __riscv_v_intrinsic>10999
    #define vreinterpret_v_u64m1_u8m1 __riscv_vreinterpret_v_u64m1_u8m1
    #define vle64_v_u64m1 __riscv_vle64_v_u64m1
    #define vle32_v_f32m1 __riscv_vle32_v_f32m1
    #define vfmv_f_s_f32m1_f32 __riscv_vfmv_f_s_f32m1_f32
    #endif
    int main(){
    	const float src[] = { 0.0f, 0.0f, 0.0f, 0.0f };
    	uint64_t ptr[2] = {0x0908060504020100, 0xFFFFFFFF0E0D0C0A};
    	vuint8m1_t a = vreinterpret_v_u64m1_u8m1(vle64_v_u64m1(ptr, 2));
    	vfloat32m1_t val = vle32_v_f32m1((const float*)(src), 4);
    	int b = (int)vfmv_f_s_f32m1_f32(val);
        return 0;
    }
   ")
  SET(ARCH_SIMD_TEST_FLAGS " -march=rv64gcv")
  SET(CMAKE_REQUIRED_FLAGS_SAVE ${CMAKE_REQUIRED_FLAGS})
  SET(CMAKE_REQUIRED_FLAGS "${ARCH_SIMD_TEST_FLAGS}") 
  CHECK_CXX_SOURCE_COMPILES("${RVV_CODE}"  COMPILE_OUT_RVV)
  SET(CMAKE_REQUIRED_FLAGS ${CMAKE_REQUIRED_FLAGS_SAVE})
  if(COMPILE_OUT_RVV)
    message("-- RVV flags were set.")
    set(CXX_RVV_FOUND TRUE)
    SET(CXX_RVV_FLAGS  "${ARCH_SIMD_TEST_FLAGS}" )
  else()
    message("-- RVV flags were NOT set.")
  endif()
  message("-- </FindRVV>")
endif()
