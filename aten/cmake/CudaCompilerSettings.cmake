if(MSVC)
  # we want to respect the standard, and we are bored of those **** .
  list(APPEND CUDA_NVCC_FLAGS "-Xcompiler /wd4819 -Xcompiler /wd4503 -Xcompiler /wd4190 -Xcompiler /wd4244 -Xcompiler /wd4251 -Xcompiler /wd4275 -Xcompiler /wd4522")
  add_definitions(-DTHC_EXPORTS)
endif()
