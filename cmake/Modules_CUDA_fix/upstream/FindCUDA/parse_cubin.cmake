#  James Bigler, NVIDIA Corp (nvidia.com - jbigler)
#  Abe Stephens, SCI Institute -- http://www.sci.utah.edu/~abe/FindCuda.html
#
#  Copyright (c) 2008 - 2009 NVIDIA Corporation.  All rights reserved.
#
#  Copyright (c) 2007-2009
#  Scientific Computing and Imaging Institute, University of Utah
#
#  This code is licensed under the MIT License.  See the FindCUDA.cmake script
#  for the text of the license.

# The MIT License
#
# License for the specific language governing rights and limitations under
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#

#######################################################################
# Parses a .cubin file produced by nvcc and reports statistics about the file.


file(READ ${input_file} file_text)

if (NOT "${file_text}" STREQUAL "")

  string(REPLACE ";" "\\;" file_text ${file_text})
  string(REPLACE "\ncode" ";code" file_text ${file_text})

  list(LENGTH file_text len)

  foreach(line ${file_text})

    # Only look at "code { }" blocks.
    if(line MATCHES "^code")

      # Break into individual lines.
      string(REGEX REPLACE "\n" ";" line ${line})

      foreach(entry ${line})

        # Extract kernel names.
        if (${entry} MATCHES "[^g]name = ([^ ]+)")
          set(entry "${CMAKE_MATCH_1}")

          # Check to see if the kernel name starts with "_"
          set(skip FALSE)
          # if (${entry} MATCHES "^_")
            # Skip the rest of this block.
            # message("Skipping ${entry}")
            # set(skip TRUE)
          # else ()
            message("Kernel:    ${entry}")
          # endif ()

        endif()

        # Skip the rest of the block if necessary
        if(NOT skip)

          # Registers
          if (${entry} MATCHES "reg([ ]+)=([ ]+)([^ ]+)")
            set(entry "${CMAKE_MATCH_3}")
            message("Registers: ${entry}")
          endif()

          # Local memory
          if (${entry} MATCHES "lmem([ ]+)=([ ]+)([^ ]+)")
            set(entry "${CMAKE_MATCH_3}")
            message("Local:     ${entry}")
          endif()

          # Shared memory
          if (${entry} MATCHES "smem([ ]+)=([ ]+)([^ ]+)")
            set(entry "${CMAKE_MATCH_3}")
            message("Shared:    ${entry}")
          endif()

          if (${entry} MATCHES "^}")
            message("")
          endif()

        endif()


      endforeach()

    endif()

  endforeach()

else()
  # message("FOUND NO DEPENDS")
endif()
