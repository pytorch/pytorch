libopencl-stub
==============

A stub opecl library that dynamically dlopen/dlsyms opencl implementations at runtime based on environment variables. Will be useful when opencl implementations are installed in non-standard paths (say pocl on android)



 LIBOPENCL_SO_PATH      -- Path to opencl so that will be searched first
 
 LIBOPENCL_SO_PATH_2    -- Searched second
 
 LIBOPENCL_SO_PATH_3    -- Searched third
 
 LIBOPENCL_SO_PATH_4    -- Searched fourth
 
 
 

Default paths will be searched otherwise

