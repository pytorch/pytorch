# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

# This script creates a list of compiled Java class files to be added to
# a jar file.  This avoids including cmake files which get created in
# the binary directory.

if (CMAKE_JAVA_CLASS_OUTPUT_PATH)
    if (EXISTS "${CMAKE_JAVA_CLASS_OUTPUT_PATH}")

        set(_JAVA_GLOBBED_FILES)
        if (CMAKE_JAR_CLASSES_PREFIX)
            foreach(JAR_CLASS_PREFIX ${CMAKE_JAR_CLASSES_PREFIX})
                message(STATUS "JAR_CLASS_PREFIX: ${JAR_CLASS_PREFIX}")

                file(GLOB_RECURSE _JAVA_GLOBBED_TMP_FILES "${CMAKE_JAVA_CLASS_OUTPUT_PATH}/${JAR_CLASS_PREFIX}/*.class")
                if (_JAVA_GLOBBED_TMP_FILES)
                    list(APPEND _JAVA_GLOBBED_FILES ${_JAVA_GLOBBED_TMP_FILES})
                endif ()
            endforeach()
        else()
            file(GLOB_RECURSE _JAVA_GLOBBED_FILES "${CMAKE_JAVA_CLASS_OUTPUT_PATH}/*.class")
        endif ()

        set(_JAVA_CLASS_FILES)
        # file(GLOB_RECURSE foo RELATIVE) is broken so we need this.
        foreach(_JAVA_GLOBBED_FILE ${_JAVA_GLOBBED_FILES})
            file(RELATIVE_PATH _JAVA_CLASS_FILE ${CMAKE_JAVA_CLASS_OUTPUT_PATH} ${_JAVA_GLOBBED_FILE})
            set(_JAVA_CLASS_FILES ${_JAVA_CLASS_FILES}${_JAVA_CLASS_FILE}\n)
        endforeach()

        # write to file
        file(WRITE ${CMAKE_JAVA_CLASS_OUTPUT_PATH}/java_class_filelist ${_JAVA_CLASS_FILES})

    else ()
        message(SEND_ERROR "FATAL: Java class output path doesn't exist")
    endif ()
else ()
    message(SEND_ERROR "FATAL: Can't find CMAKE_JAVA_CLASS_OUTPUT_PATH")
endif ()
