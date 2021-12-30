if(TARGET caffe2::Threads)
  return()
endif()

find_package(Threads REQUIRED)

# Threads::Threads doesn't work if the target has CUDA code
if(THREADS_FOUND)
  add_library(caffe2::Threads INTERFACE IMPORTED)

  if(THREADS_HAVE_PTHREAD_ARG)
    set(compile_options
        $<$<COMPILE_LANGUAGE:C>:-pthread>
        $<$<COMPILE_LANGUAGE:CXX>:-pthread>)
    if(USE_CUDA)
      list(APPEND compile_options
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler -pthread>)
    endif()

    set_property(TARGET caffe2::Threads
                 PROPERTY INTERFACE_COMPILE_OPTIONS
                 ${compile_options})
  endif()

  if(CMAKE_THREAD_LIBS_INIT)
    set_property(TARGET caffe2::Threads
                 PROPERTY INTERFACE_LINK_LIBRARIES "${CMAKE_THREAD_LIBS_INIT}")
  endif()
endif()
