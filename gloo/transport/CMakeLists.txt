set(GLOO_TRANSPORT_SRCS
  "${CMAKE_CURRENT_SOURCE_DIR}/address.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/buffer.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/device.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/pair.cc"
  )

add_subdirectory(tcp)
if(USE_IBVERBS)
    add_subdirectory(ibverbs)
endif()

list(APPEND GLOO_SRCS ${GLOO_TRANSPORT_SRCS})
set(GLOO_SRCS ${GLOO_SRCS} PARENT_SCOPE)
