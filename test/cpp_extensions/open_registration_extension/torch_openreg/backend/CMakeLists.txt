file(GLOB_RECURSE SOURCE_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)

add_library(openreg SHARED ${SOURCE_FILES})
target_include_directories(openreg
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
)
install(TARGETS openreg LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
