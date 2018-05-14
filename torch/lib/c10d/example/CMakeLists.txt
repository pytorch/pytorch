add_executable(allreduce allreduce.cpp)
target_include_directories(allreduce PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_link_libraries(allreduce store process_group process_group_gloo pthread gloo)
