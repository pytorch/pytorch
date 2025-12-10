# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.


# This file is included in CMakeSystemSpecificInformation.cmake if
# the Sublime Text 2 extra generator has been selected.

find_program(CMAKE_SUBLIMETEXT_EXECUTABLE
    NAMES subl3 subl sublime_text
    PATHS
        "/Applications/Sublime Text.app/Contents/SharedSupport/bin"
        "/Applications/Sublime Text 3.app/Contents/SharedSupport/bin"
        "/Applications/Sublime Text 2.app/Contents/SharedSupport/bin"
        "$ENV{HOME}/Applications/Sublime Text.app/Contents/SharedSupport/bin"
        "$ENV{HOME}/Applications/Sublime Text 3.app/Contents/SharedSupport/bin"
        "$ENV{HOME}/Applications/Sublime Text 2.app/Contents/SharedSupport/bin"
        "/opt/sublime_text"
        "/opt/sublime_text_3"
    DOC "The Sublime Text executable")

if(CMAKE_SUBLIMETEXT_EXECUTABLE)
  set(CMAKE_OPEN_PROJECT_COMMAND "${CMAKE_SUBLIMETEXT_EXECUTABLE} --project <PROJECT_FILE>" )
endif()
