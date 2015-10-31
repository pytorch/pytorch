FILE(GLOB snippets_SRCS "*.cpp")

ADD_CUSTOM_TARGET(unsupported_snippets)

FOREACH(snippet_src ${snippets_SRCS})
  GET_FILENAME_COMPONENT(snippet ${snippet_src} NAME_WE)
  SET(compile_snippet_target compile_${snippet})
  SET(compile_snippet_src ${compile_snippet_target}.cpp)
  FILE(READ ${snippet_src} snippet_source_code)
  CONFIGURE_FILE(${PROJECT_SOURCE_DIR}/doc/snippets/compile_snippet.cpp.in
                 ${CMAKE_CURRENT_BINARY_DIR}/${compile_snippet_src})
  ADD_EXECUTABLE(${compile_snippet_target}
                 ${CMAKE_CURRENT_BINARY_DIR}/${compile_snippet_src})
  if(EIGEN_STANDARD_LIBRARIES_TO_LINK_TO)
    target_link_libraries(${compile_snippet_target} ${EIGEN_STANDARD_LIBRARIES_TO_LINK_TO})
  endif()
  ADD_CUSTOM_COMMAND(
    TARGET ${compile_snippet_target}
    POST_BUILD
    COMMAND ${compile_snippet_target}
    ARGS >${CMAKE_CURRENT_BINARY_DIR}/${snippet}.out
  )
  ADD_DEPENDENCIES(unsupported_snippets ${compile_snippet_target})
  set_source_files_properties(${CMAKE_CURRENT_BINARY_DIR}/${compile_snippet_src}
                              PROPERTIES OBJECT_DEPENDS ${snippet_src})
ENDFOREACH(snippet_src)
