find_package(Git)

# Extract a version from the latest tag matching something like v1.2.3.4
function(get_git_version var_numeric var dir)
    if(NOT DEFINED ${var})
        set(version_numeric "0.0.0")
        set(version "0.0.0-unknown")
        if(GIT_EXECUTABLE)
            set(command
                "${GIT_EXECUTABLE}"
                -C
                "${dir}"
                describe
                --tags
                --match
                v*
            )
            execute_process(
                COMMAND ${command}
                RESULT_VARIABLE result
                OUTPUT_STRIP_TRAILING_WHITESPACE
                OUTPUT_VARIABLE version_out
            )
            if(NOT result EQUAL 0)
                message(
                    WARNING
                    "Getting ${var} failed: ${command} returned ${result}\nIs this a Git repo with tags?\nConsider settings -D${var} to specify a version manually"
                )
            elseif("${version_out}" MATCHES "^v(([0-9]+(\\.[0-9]+)*).*)")
                set(version "${CMAKE_MATCH_1}")
                set(version_numeric "${CMAKE_MATCH_2}")
            else()
                message(
                    WARNING
                    "Couldn't parse version (like v1.2.3 or v1.2.3-foo) from ${version_out}, using ${version} for now"
                )
            endif()
        else()
            message(
                WARNING
                "Couldn't find git executable to get ${var}, please use -D${var}, using ${version} for now"
            )
        endif()
    endif()

    set(${var_numeric}
        ${version_numeric}
        CACHE STRING
        "The project version numeric part, detected using git if available"
    )
    set(${var}
        ${version}
        CACHE STRING
        "The project version, detected using git if available"
    )
endfunction()
