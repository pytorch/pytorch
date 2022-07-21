if(NOT __UCC_INCLUDED)
  set(__UCC_INCLUDED TRUE)

  if(USE_SYSTEM_UCC)
    set(UCX_HOME $ENV{UCX_HOME} CACHE PATH "UCX install directory")
    set(UCC_HOME $ENV{UCC_HOME} CACHE PATH "UCC install directory")

    add_library(__caffe2_ucc INTERFACE)

    target_include_directories(__caffe2_ucc INTERFACE ${UCX_HOME}/include/)
    target_include_directories(__caffe2_ucc INTERFACE ${UCC_HOME}/include/)

    target_link_libraries(__caffe2_ucc INTERFACE ${UCX_HOME}/lib/libucp.so)
    target_link_libraries(__caffe2_ucc INTERFACE ${UCX_HOME}/lib/libucs.so)
    target_link_libraries(__caffe2_ucc INTERFACE ${UCC_HOME}/lib/libucc.so)
  else()
    message(FATAL_ERROR "USE_SYSTEM_UCC=OFF is not supported yet when using UCC")
  endif()

endif()
