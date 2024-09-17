# Check if the processor is an ARM and if Neon instruction are available on the machine where
# the project is compiled.

IF(CMAKE_SYSTEM_NAME MATCHES "Linux")
   EXECUTE_PROCESS(COMMAND cat /proc/cpuinfo OUTPUT_VARIABLE CPUINFO)

   #neon instruction can be found on the majority part of modern ARM processor
   STRING(REGEX REPLACE "^.*(neon).*$" "\\1" NEON_THERE "${CPUINFO}")
   STRING(COMPARE EQUAL "neon" "${NEON_THERE}" NEON_TRUE)
   IF (NEON_TRUE)
      set(NEON_FOUND true CACHE BOOL "NEON available on host")
   ELSE (NEON_TRUE)
      set(NEON_FOUND false CACHE BOOL "NEON available on host")
   ENDIF (NEON_TRUE)

   # on ARMv8, neon is inherit and instead listed as 'asimd' in /proc/cpuinfo
   STRING(REGEX REPLACE "^.*(asimd).*$" "\\1" ASIMD_THERE "${CPUINFO}")
   STRING(COMPARE EQUAL "asimd" "${ASIMD_THERE}" ASIMD_TRUE)
   IF (ASIMD_TRUE)
      set(ASIMD_FOUND true CACHE BOOL "ASIMD/NEON available on host")
   ELSE (ASIMD_TRUE)
      set(ASIMD_FOUND false CACHE BOOL "ASIMD/NEON available on host")
   ENDIF (ASIMD_TRUE)

   #Find the processor type (for now OMAP3 or OMAP4)
   STRING(REGEX REPLACE "^.*(OMAP3).*$" "\\1" OMAP3_THERE "${CPUINFO}")
   STRING(COMPARE EQUAL "OMAP3" "${OMAP3_THERE}" OMAP3_TRUE)
   IF (OMAP3_TRUE)
      set(CORTEXA8_FOUND true CACHE BOOL "OMAP3 available on host")
   ELSE (OMAP3_TRUE)
      set(CORTEXA8_FOUND false CACHE BOOL "OMAP3 available on host")
   ENDIF (OMAP3_TRUE)

   #Find the processor type (for now OMAP3 or OMAP4)
   STRING(REGEX REPLACE "^.*(OMAP4).*$" "\\1" OMAP4_THERE "${CPUINFO}")
   STRING(COMPARE EQUAL "OMAP4" "${OMAP4_THERE}" OMAP4_TRUE)
   IF (OMAP4_TRUE)
      set(CORTEXA9_FOUND true CACHE BOOL "OMAP4 available on host")
   ELSE (OMAP4_TRUE)
      set(CORTEXA9_FOUND false CACHE BOOL "OMAP4 available on host")
   ENDIF (OMAP4_TRUE)

ELSEIF(CMAKE_SYSTEM_NAME MATCHES "Darwin")
   IF(CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64" AND NOT CMAKE_OSX_ARCHITECTURES STREQUAL "x86_64")
      set(NEON_FOUND true CACHE BOOL "NEON available on ARM64")
   ENDIF()
   EXECUTE_PROCESS(COMMAND /usr/sbin/sysctl -n machdep.cpu.features OUTPUT_VARIABLE
      CPUINFO)

   IF(NOT CPUINFO STREQUAL "")
       #neon instruction can be found on the majority part of modern ARM processor
       STRING(REGEX REPLACE "^.*(neon).*$" "\\1" NEON_THERE "${CPUINFO}")
       STRING(COMPARE EQUAL "neon" "${NEON_THERE}" NEON_TRUE)
       IF (NEON_TRUE)
          set(NEON_FOUND true CACHE BOOL "NEON available on host")
       ELSE (NEON_TRUE)
          set(NEON_FOUND false CACHE BOOL "NEON available on host")
       ENDIF (NEON_TRUE)
   ENDIF()

ELSEIF(CMAKE_SYSTEM_NAME MATCHES "Windows")
   # TODO
   set(CORTEXA8_FOUND   false CACHE BOOL "OMAP3 not available on host")
   set(CORTEXA9_FOUND   false CACHE BOOL "OMAP4 not available on host")
   set(NEON_FOUND   false CACHE BOOL "NEON not available on host")
ELSE(CMAKE_SYSTEM_NAME MATCHES "Linux")
   set(CORTEXA8_FOUND   false CACHE BOOL "OMAP3 not available on host")
   set(CORTEXA9_FOUND   false CACHE BOOL "OMAP4 not available on host")
   set(NEON_FOUND   false CACHE BOOL "NEON not available on host")
ENDIF(CMAKE_SYSTEM_NAME MATCHES "Linux")

if(NOT NEON_FOUND)
      MESSAGE(STATUS "Could not find hardware support for NEON on this machine.")
endif(NOT NEON_FOUND)
if(NOT CORTEXA8_FOUND)
      MESSAGE(STATUS "No OMAP3 processor on this machine.")
endif(NOT CORTEXA8_FOUND)
if(NOT CORTEXA9_FOUND)
      MESSAGE(STATUS "No OMAP4 processor on this machine.")
endif(NOT CORTEXA9_FOUND)
mark_as_advanced(NEON_FOUND)
