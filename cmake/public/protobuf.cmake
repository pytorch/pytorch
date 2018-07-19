# ---[ Protobuf

# We will try to use the config mode first, and then manual find.
find_package(Protobuf CONFIG QUIET)
if (NOT Protobuf_FOUND)
  find_package(Protobuf MODULE QUIET)
endif()

if ((TARGET protobuf::libprotobuf OR TARGET protobuf::libprotobuf-lite) AND TARGET protobuf::protoc)
  # Hooray. This is the most ideal situation, meaning that you either have a
  # Protobuf config file installed (like on Windows), or you are using a
  # modern CMake that ships with a FindProtobuf.cmake file that produces
  # modern targets.
  message(STATUS "Caffe2: Found protobuf with new-style protobuf targets.")
elseif(Protobuf_FOUND OR PROTOBUF_FOUND)
  # If the modern targets are not present, we will generate them for you for
  # backward compatibility. This is backported from CMake's new FindProtobuf.cmake
  # content.
  if ((NOT PROTOBUF_LIBRARY) AND (NOT PROTOBUF_LITE_LIBRARY))
    message(FATAL_ERROR
        "Caffe2: Found protobuf with old style targets, but could not find targets."
        " PROTOBUF_LIBRARY: " ${PROTOBUF_LIBRARY}
        " PROTOBUF_LITE_LIBRARY: " ${PROTOBUF_LITE_LIBRARY}
        " Protobuf_LIBRARY: " ${Protobuf_LIBRARY}
        " Protobuf_LITE_LIBRARY: " ${Protobuf_LITE_LIBRARY})
  endif()
  message(STATUS "Caffe2: Found protobuf with old-style protobuf targets.")

  if(PROTOBUF_LIBRARY)
    if (NOT TARGET protobuf::libprotobuf)
      add_library(protobuf::libprotobuf UNKNOWN IMPORTED)
      set_target_properties(protobuf::libprotobuf PROPERTIES
          INTERFACE_INCLUDE_DIRECTORIES "${PROTOBUF_INCLUDE_DIRS}")
    endif()
    if(EXISTS "${PROTOBUF_LIBRARY}")
      set_target_properties(protobuf::libprotobuf PROPERTIES
          IMPORTED_LOCATION "${PROTOBUF_LIBRARY}")
    endif()
    if(EXISTS "${PROTOBUF_LIBRARY_RELEASE}")
      set_property(TARGET protobuf::libprotobuf APPEND PROPERTY
          IMPORTED_CONFIGURATIONS RELEASE)
      set_target_properties(protobuf::libprotobuf PROPERTIES
          IMPORTED_LOCATION_RELEASE "${PROTOBUF_LIBRARY_RELEASE}")
    endif()
    if(EXISTS "${PROTOBUF_LIBRARY_DEBUG}")
      set_property(TARGET protobuf::libprotobuf APPEND PROPERTY
          IMPORTED_CONFIGURATIONS DEBUG)
      set_target_properties(protobuf::libprotobuf PROPERTIES
          IMPORTED_LOCATION_DEBUG "${PROTOBUF_LIBRARY_DEBUG}")
    endif()
  endif()

  if(PROTOBUF_LITE_LIBRARY)
    if (NOT TARGET protobuf::libprotobuf-lite)
      add_library(protobuf::libprotobuf-lite UNKNOWN IMPORTED)
      set_target_properties(protobuf::libprotobuf-lite PROPERTIES
          INTERFACE_INCLUDE_DIRECTORIES "${PROTOBUF_INCLUDE_DIRS}")
    endif()
    if(EXISTS "${PROTOBUF_LITE_LIBRARY}")
      set_target_properties(protobuf::libprotobuf-lite PROPERTIES
          IMPORTED_LOCATION "${PROTOBUF_LITE_LIBRARY}")
    endif()
    if(EXISTS "${PROTOBUF_LITE_LIBRARY_RELEASE}")
      set_property(TARGET protobuf::libprotobuf-lite APPEND PROPERTY
          IMPORTED_CONFIGURATIONS RELEASE)
      set_target_properties(protobuf::libprotobuf-lite PROPERTIES
          IMPORTED_LOCATION_RELEASE "${PROTOBUF_LITE_LIBRARY_RELEASE}")
    endif()
    if(EXISTS "${PROTOBUF_LITE_LIBRARY_DEBUG}")
      set_property(TARGET protobuf::libprotobuf-lite APPEND PROPERTY
          IMPORTED_CONFIGURATIONS DEBUG)
      set_target_properties(protobuf::libprotobuf-lite PROPERTIES
          IMPORTED_LOCATION_DEBUG "${PROTOBUF_LITE_LIBRARY_DEBUG}")
    endif()
  endif()

  if(PROTOBUF_PROTOC_EXECUTABLE)
    if (NOT TARGET protobuf::protoc)
      add_executable(protobuf::protoc IMPORTED)
    endif()
    set_property(TARGET protobuf::protoc PROPERTY
        IMPORTED_LOCATION ${PROTOBUF_PROTOC_EXECUTABLE})
  endif()
endif()

# After above, we should have the protobuf related target now.
if ((NOT TARGET protobuf::libprotobuf) AND (NOT TARGET protobuf::libprotobuf-lite))
  message(WARNING
      "Protobuf cannot be found. Depending on whether you are building Caffe2 "
      "or a Caffe2 dependent library, the next warning / error will give you "
      "more info.")
endif()
