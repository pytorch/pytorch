set(test_srcs
  FileStoreTest.cpp
  )

set(test_libraries
  store
  pthread
  )

foreach(test_src ${test_srcs})
  get_filename_component(test_name ${test_src} NAME_WE)
  add_executable(${test_name} "${test_src}")
  target_include_directories(${test_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
  target_link_libraries(${test_name} ${test_libraries})
  add_test(NAME ${test_name} COMMAND $<TARGET_FILE:${test_name}>)
endforeach()
