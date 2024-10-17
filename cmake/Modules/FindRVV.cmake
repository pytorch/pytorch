IF(CMAKE_SYSTEM_NAME MATCHES "Linux")
  message("-- <FindRVV>")
  SET(RVV_CODE "
    #include <riscv_vector.h>
    int main(){
        const float src[] = { 0.0f, 0.0f, 0.0f, 0.0f };
        uint64_t ptr[2] = {0x0908060504020100, 0xFFFFFFFF0E0D0C0A};
        vuint8m1_t a = __riscv_vreinterpret_v_u64m1_u8m1(__riscv_vle64_v_u64m1(ptr, 2));
        vfloat32m1_t val = __riscv_vle32_v_f32m1((const float*)(src), 4);
        int b = (int)__riscv_vfmv_f_s_f32m1_f32(val);
        return 0;
    }
   ")
  SET(READ_VECTOR_LENGTH_CODE "
    #include <riscv_vector.h>
    #include <iostream>
    int main(){
        unsigned long vlen_bytes = __riscv_vlenb();
        unsigned long vlen_bits = vlen_bytes * 8;
        std::cout << vlen_bits << std::endl;
        return 0;
    }
   ")
  file(WRITE ${CMAKE_BINARY_DIR}/read_vector_length.cpp "${READ_VECTOR_LENGTH_CODE}")

  SET(ARCH_SIMD_TEST_FLAGS "-march=rv64gcv")
  SET(CMAKE_REQUIRED_FLAGS_SAVE ${CMAKE_REQUIRED_FLAGS})
  SET(CMAKE_REQUIRED_FLAGS "${ARCH_SIMD_TEST_FLAGS}")
  CHECK_CXX_SOURCE_COMPILES("${RVV_CODE}"  COMPILE_OUT_RVV)
  SET(CMAKE_REQUIRED_FLAGS ${CMAKE_REQUIRED_FLAGS_SAVE})
  if(COMPILE_OUT_RVV)
    execute_process(
      COMMAND
      "${CMAKE_CXX_COMPILER}"
      "${ARCH_SIMD_TEST_FLAGS}"
      "${CMAKE_BINARY_DIR}/read_vector_length.cpp"
      "-o"
      "${CMAKE_BINARY_DIR}/read_vector_length"
      RESULT_VARIABLE VECTOR_LENGTH_CHECK_COMPILE_RESULT)
    if(VECTOR_LENGTH_CHECK_COMPILE_RESULT)
      message(FATAL_ERROR "Could not compile RISC-V Vector Length Check: ${VECTOR_LENGTH_CHECK_COMPILE_RESULT}")
    endif()
    execute_process(
      COMMAND "${CMAKE_BINARY_DIR}/read_vector_length"
      RESULT_VARIABLE VECTOR_LENGTH_CHECK_RESULT
      OUTPUT_VARIABLE VLEN_BITS
      OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(VECTOR_LENGTH_CHECK_RESULT)
      message(WARNING "Could not run RISC-V Vector Length Check: ${VECTOR_LENGTH_CHECK_RESULT}")
    endif()
    message("-- RVV flags were set.")
    message("-- RISC-V CPU Vector Length: ${VLEN_BITS} bits")
    set(CXX_RVV_FOUND TRUE)
    SET(CXX_RVV_FLAGS  "${ARCH_SIMD_TEST_FLAGS} -mrvv-vector-bits=${VLEN_BITS}" )
  else()
    message("-- RVV flags were NOT set.")
  endif()
  message("-- </FindRVV>")
endif()
