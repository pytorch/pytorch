check_include_file(cpuid.h HAVE_CPUID_H)

# Check for a cpuid intrinsic
if(HAVE_CPUID_H)
    CHECK_C_SOURCE_COMPILES("#include <cpuid.h>
        int main()
        {
            unsigned int eax, ebx, ecx, edx;
            return __get_cpuid(0, &eax, &ebx, &ecx, &edx);
        }" HAVE_GCC_GET_CPUID)
endif()

if(HAVE_GCC_GET_CPUID)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DHAVE_GCC_GET_CPUID")
endif()
