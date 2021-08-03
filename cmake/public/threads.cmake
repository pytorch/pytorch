if (TARGET caffe2::Threads)
  return()
endif()

find_package(Threads REQUIRED)

# Threads::Threads doesn't work if the target has CUDA code
if(THREADS_FOUND)
  add_library(caffe2::Threads INTERFACE IMPORTED)

  if(THREADS_HAVE_PTHREAD_ARG)
    set_property(TARGET caffe2::Threads
                 PROPERTY INTERFACE_COMPILE_OPTIONS
                   $<$<COMPILE_LANGUAGE:CXX>:-pthread>
                   $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler -pthread>)
  endif()

  if(CMAKE_THREAD_LIBS_INIT)
    set_property(TARGET caffe2::Threads
                 PROPERTY INTERFACE_LINK_LIBRARIES "${CMAKE_THREAD_LIBS_INIT}")
  endif()
endif()
