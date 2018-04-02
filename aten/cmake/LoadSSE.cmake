# - Finds SSE
# - Sets C/CXX flags to enable SSE compilation

find_package(SSE) # checks SSE, AVX and AVX2

if(C_SSE2_FOUND)
  message(STATUS "SSE2 Found")
  set(CMAKE_C_FLAGS "${C_SSE2_FLAGS} -DUSE_SSE2 ${CMAKE_C_FLAGS}")
endif()

if(C_SSE4_1_FOUND AND C_SSE4_2_FOUND)
  set(CMAKE_C_FLAGS "${C_SSE4_1_FLAGS} -DUSE_SSE4_1 ${C_SSE4_2_FLAGS} -DUSE_SSE4_2 ${CMAKE_C_FLAGS}")
endif()

if(C_SSE3_FOUND)
  message(STATUS "SSE3 Found")
  set(CMAKE_C_FLAGS "${C_SSE3_FLAGS} -DUSE_SSE3 ${CMAKE_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${C_SSE3_FLAGS} -DUSE_SSE3 ${CMAKE_CXX_FLAGS}")
endif()

# we don't set -mavx and -mavx2 flags globally, but only for specific files
# however, we want to enable the AVX codepaths, so we still need to
# add USE_AVX and USE_AVX2 macro defines
if(C_AVX_FOUND)
  message(STATUS "AVX Found")
  set(CMAKE_C_FLAGS "-DUSE_AVX ${CMAKE_C_FLAGS}")
endif()

if(C_AVX2_FOUND)
  message(STATUS "AVX2 Found")
  set(CMAKE_C_FLAGS "-DUSE_AVX2 ${CMAKE_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "-DUSE_AVX2 ${CMAKE_CXX_FLAGS}")
endif()
