add_executable(allreduce allreduce.cpp)
target_include_directories(allreduce PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_link_libraries(allreduce pthread c10d)
