# ---[ mtia

# Poor man's include guard
if(TARGET torch::mtiart)
  return()
endif()

set(PYTORCH_FOUND_MTIA TRUE)

# mtiart
add_library(torch::mtiart INTERFACE IMPORTED)
