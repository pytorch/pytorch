find_package(MAGMA)
if(CUDA_FOUND AND MAGMA_FOUND)
  include_directories("${MAGMA_INCLUDE_DIR}")
  set(CMAKE_REQUIRED_INCLUDES "${MAGMA_INCLUDE_DIR};${CUDA_INCLUDE_DIRS}")
  include(CheckPrototypeDefinition)
  check_prototype_definition(magma_get_sgeqrf_nb
   "magma_int_t magma_get_sgeqrf_nb( magma_int_t m, magma_int_t n );"
   "0"
   "magma.h"
    MAGMA_V2)
  if(MAGMA_V2)
    add_definitions(-DMAGMA_V2)
  endif()

  set(USE_MAGMA 1)
  message(STATUS "Compiling with MAGMA support")
  message(STATUS "MAGMA INCLUDE DIRECTORIES: ${MAGMA_INCLUDE_DIR}")
  message(STATUS "MAGMA LIBRARIES: ${MAGMA_LIBRARIES}")
  message(STATUS "MAGMA V2 check: ${MAGMA_V2}")
else()
  message(STATUS "MAGMA not found. Compiling without MAGMA support")
endif()
