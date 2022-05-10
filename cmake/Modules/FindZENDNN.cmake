IF (NOT ZENDNN_FOUND)

file(GLOB zendnn_src_common_cpp "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ZenDNN/src/common/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ZenDNN/src/cpu/*.cpp"
"${CMAKE_CURRENT_SOURCE_DIR}/third_party/ZenDNN/src/cpu/gemm/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ZenDNN/src/cpu/gemm/f32/*.cpp"
"${CMAKE_CURRENT_SOURCE_DIR}/third_party/ZenDNN/src/cpu/gemm/s8x8s32/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ZenDNN/src/cpu/matmul/*.cpp"
"${CMAKE_CURRENT_SOURCE_DIR}/third_party/ZenDNN/src/cpu/reorder/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ZenDNN/src/cpu/rnn/*.cpp"
"${CMAKE_CURRENT_SOURCE_DIR}/third_party/ZenDNN/src/cpu/x64/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ZenDNN/src/cpu/x64/brgemm/*.cpp"
"${CMAKE_CURRENT_SOURCE_DIR}/third_party/ZenDNN/src/cpu/x64/gemm/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ZenDNN/src/cpu/x64/gemm/amx/*.cpp"
"${CMAKE_CURRENT_SOURCE_DIR}/third_party/ZenDNN/src/cpu/x64/gemm/bf16/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ZenDNN/src/cpu/x64/gemm/f32/*.cpp"
"${CMAKE_CURRENT_SOURCE_DIR}/third_party/ZenDNN/src/cpu/x64/gemm/s8x8s32/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ZenDNN/src/cpu/x64/injectors/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ZenDNN/src/cpu/x64/lrn/*.cpp"
"${CMAKE_CURRENT_SOURCE_DIR}/third_party/ZenDNN/src/cpu/x64/matmul/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ZenDNN/src/cpu/x64/prelu/*.cpp"
"${CMAKE_CURRENT_SOURCE_DIR}/third_party/ZenDNN/src/cpu/x64/rnn/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ZenDNN/src/cpu/x64/shuffle/*.cpp")

set(GENERATED_CXX_ZEN
    ${zendnn_src_common_cpp}
  )

IF(CMAKE_BUILD_TYPE STREQUAL "Debug")
   SET(BUILD_FLAG 0)
ELSE()
   SET(BUILD_FLAG 1)
ENDIF(CMAKE_BUILD_TYPE STREQUAL "Debug")

add_custom_target(libamdZenDNN ALL
    DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/build/lib/libamdZenDNN.a
)

add_custom_command(
   OUTPUT
        ${CMAKE_CURRENT_SOURCE_DIR}/build/lib/libamdZenDNN.a
   WORKING_DIRECTORY
       ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ZenDNN
   COMMAND
       make -j ZENDNN_BLIS_PATH?=${CMAKE_CURRENT_SOURCE_DIR}/build/blis_gcc_build AOCC=0 ARCHIVE=1 RELEASE=${BUILD_FLAG}
   COMMAND
       cp _out/lib/libamdZenDNN.a ${CMAKE_CURRENT_SOURCE_DIR}/build/lib
   DEPENDS
        ${zendnn_src_common_cpp}
   COMMAND
        make clean
)

add_dependencies(libamdZenDNN libamdblis)

SET(ZENDNN_INCLUDE_SEARCH_PATHS
 ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ZenDNN/inc
)

FIND_PATH(ZENDNN_INCLUDE_DIR NAMES zendnn_config.h zendnn.h zendnn_types.h zendnn_debug.h zendnn_version.h PATHS ${ZENDNN_INCLUDE_SEARCH_PATHS})
IF(NOT ZENDNN_INCLUDE_DIR)
    MESSAGE(STATUS "Could not find ZENDNN include.")
    RETURN()
ENDIF(NOT ZENDNN_INCLUDE_DIR)

SET(ZENDNN_LIB_SEARCH_PATHS
 ${CMAKE_CURRENT_SOURCE_DIR}/build/lib
)

LIST(APPEND ZENDNN_LIBRARIES ${ZENDNN_LIB_SEARCH_PATHS}/libamdZenDNN.a)

add_custom_target(libamdblis ALL
    DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/build/lib/libblis-mt.a
)

add_custom_command(
    OUTPUT
       ${CMAKE_CURRENT_SOURCE_DIR}/build/lib/libblis-mt.a
   WORKING_DIRECTORY
       ${CMAKE_CURRENT_SOURCE_DIR}/third_party/blis
   COMMAND
       make clean && make distclean && CC=gcc  ./configure --prefix=${CMAKE_CURRENT_SOURCE_DIR}/build/blis_gcc_build  --enable-threading=openmp --enable-cblas zen3 && make -j install
   COMMAND
       cd ${CMAKE_CURRENT_SOURCE_DIR}/build
   COMMAND
       cp blis_gcc_build/lib/libblis-mt.a ${CMAKE_CURRENT_SOURCE_DIR}/build/lib
   COMMAND
       cp -r blis_gcc_build/include/blis/* blis_gcc_build/include
)

LIST(APPEND ZENDNN_LIBRARIES ${CMAKE_CURRENT_SOURCE_DIR}/build/lib/libblis-mt.a)

MARK_AS_ADVANCED(
    ZENDNN_INCLUDE_DIR
    ZENDNN_LIBRARIES
        amdZenDNN
)

#
IF(NOT USE_OPENMP)
    MESSAGE(FATAL_ERROR "ZenDNN requires OMP library")
    RETURN()
ENDIF()

IF(USE_TBB)
  MESSAGE(FATAL_ERROR "ZenDNN requires blis library, set USE_TBB=0")
  RETURN()
ENDIF(USE_TBB)


SET(ZENDNN_FOUND ON)
IF (ZENDNN_FOUND)
    IF (NOT ZENDNN_FIND_QUIETLY)
        MESSAGE(STATUS "Found ZENDNN libraries: ${ZENDNN_LIBRARIES}")
        MESSAGE(STATUS "Found ZENDNN include: ${ZENDNN_INCLUDE_DIR}")
    ENDIF (NOT ZENDNN_FIND_QUIETLY)
ELSE (ZENDNN_FOUND)
    IF (ZENDNN_FIND_REQUIRED)
        MESSAGE(FATAL_ERROR "Could not find ZENDNN")
    ENDIF (ZENDNN_FIND_REQUIRED)
ENDIF (ZENDNN_FOUND)

ENDIF (NOT ZENDNN_FOUND)
