FILE(GLOB examples_SRCS "*.cpp")

ADD_CUSTOM_TARGET(unsupported_examples)

INCLUDE_DIRECTORIES(../../../unsupported ../../../unsupported/test)

FOREACH(example_src ${examples_SRCS})
  GET_FILENAME_COMPONENT(example ${example_src} NAME_WE)
  ADD_EXECUTABLE(example_${example} ${example_src})
  if(EIGEN_STANDARD_LIBRARIES_TO_LINK_TO)
    target_link_libraries(example_${example} ${EIGEN_STANDARD_LIBRARIES_TO_LINK_TO})
  endif()
  ADD_CUSTOM_COMMAND(
    TARGET example_${example}
    POST_BUILD
    COMMAND example_${example}
    ARGS >${CMAKE_CURRENT_BINARY_DIR}/${example}.out
  )
  ADD_DEPENDENCIES(unsupported_examples example_${example})
ENDFOREACH(example_src)
