if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    message(WARNING "Gloo doesn't support OSX (uses epoll)")
else()
    list(APPEND GLOO_TRANSPORT_SRCS
      "${CMAKE_CURRENT_SOURCE_DIR}/address.cc"
      "${CMAKE_CURRENT_SOURCE_DIR}/buffer.cc"
      "${CMAKE_CURRENT_SOURCE_DIR}/device.cc"
      "${CMAKE_CURRENT_SOURCE_DIR}/pair.cc"
      )
endif()

set(GLOO_TRANSPORT_SRCS ${GLOO_TRANSPORT_SRCS} PARENT_SCOPE)
