# ---[ absl logging

find_package(absl CONFIG QUIET)
if(NOT TARGET absl::log)
  find_package(absl MODULE QUIET)
endif()

if(TARGET absl::log)
  message(STATUS "Caffe2: Found absl with new-style absl target.")
endif()

if(NOT TARGET absl::log)
  message(WARNING
      "Caffe2: absl cannot be found. Depending on whether you are building "
      "Caffe2 or a Caffe2 dependent library, the next warning / error will "
      "give you more info.")
endif()
