# TODO: Which versions of GCC does this fail on?
CHECK_C_SOURCE_COMPILES("#include <stdint.h>
    static inline void cpuid(uint32_t *eax, uint32_t *ebx,
    			 uint32_t *ecx, uint32_t *edx)
    {
      uint32_t a = *eax, b, c = *ecx, d;
      asm volatile ( \"cpuid\" : \"+a\"(a), \"=b\"(b), \"+c\"(c), \"=d\"(d) );
      *eax = a; *ebx = b; *ecx = c; *edx = d;
    }
    int main() {
      uint32_t a,b,c,d;
      cpuid(&a, &b, &c, &d);
      return 0;
    }" NO_GCC_EBX_FPIC_BUG)

if(NOT NO_GCC_EBX_FPIC_BUG)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DUSE_GCC_GET_CPUID")
endif()
