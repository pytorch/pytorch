function(check_assets_for_file json_content filename found_var)
    string(JSON asset_count LENGTH "${json_content}" "assets")
    set(found "FALSE")

    # Never change, CMake...
    math(EXPR max_asset_index "${asset_count} - 1")
    foreach(i RANGE 0 ${max_asset_index})
        string(JSON asset_name GET "${json_content}" "assets" ${i} "name")
        if("${asset_name}" STREQUAL "${filename}")
            set(found "TRUE")
            break()
        endif()
    endforeach()
    set(${found_var} "${found}" PARENT_SCOPE)
endfunction()

function(
    get_latest
    owner
    repo
    os
    arch
    github_token
    out_var
)
    set(json_output_file
        "${CMAKE_CURRENT_BINARY_DIR}/${owner}_${repo}_release_info.json"
    )
    set(latest_release_url
        "https://api.github.com/repos/${owner}/${repo}/releases/latest"
    )

    set(download_args
        "${latest_release_url}"
        "${json_output_file}"
        STATUS
        download_statuses
    )

    if(github_token)
        list(
            APPEND
            download_args
            HTTPHEADER
            "Authorization: token ${github_token}"
        )
    endif()

    file(DOWNLOAD ${download_args})
    list(GET download_statuses 0 status_code)
    if(NOT status_code EQUAL 0)
        message(
            WARNING
            "Failed to download latest release info from ${latest_release_url}"
        )
        return()
    endif()

    # Get the tag from this release json file
    file(READ "${json_output_file}" latest_json_content)
    string(JSON latest_release_tag GET "${latest_json_content}" "tag_name")
    string(REGEX REPLACE "^v" "" latest_version "${latest_release_tag}")

    # Check if the expected ZIP file is in the latest release
    set(desired_zip "${repo}-${latest_version}-${os}-${arch}.zip")
    message(
        VERBOSE
        "searching for the prebuilt slang-llvm library in ${latest_release_url}"
    )
    check_assets_for_file(
        "${latest_json_content}"
        "${desired_zip}"
        file_found_latest
    )

    if(file_found_latest)
        # If we got it, we found a good version
        set(${out_var} "${latest_version}" PARENT_SCOPE)
    else()
        message(
            WARNING
            "No release binary for ${os}-${arch} exists for the latest version: ${latest_version}"
        )
    endif()
endfunction()

function(
    check_release_and_get_latest
    owner
    repo
    version
    os
    arch
    github_token
    out_var
)
    # Construct the URL for the specified version's release API endpoint
    set(version_url
        "https://api.github.com/repos/${owner}/${repo}/releases/tags/v${version}"
    )

    set(json_output_file
        "${CMAKE_CURRENT_BINARY_DIR}/${owner}_${repo}_release_info.json"
    )

    # Prepare download arguments
    set(download_args
        "${version_url}"
        "${json_output_file}"
        STATUS
        download_statuses
    )

    if(github_token)
        # Add authorization header if token is provided
        list(
            APPEND
            download_args
            HTTPHEADER
            "Authorization: token ${github_token}"
        )
    endif()

    # Perform the download
    file(DOWNLOAD ${download_args})

    # Check if the downloading was successful
    list(GET download_statuses 0 status_code)
    if(status_code EQUAL 0)
        file(READ "${json_output_file}" json_content)

        # Check if the specified version contains the expected ZIP file
        set(desired_zip "${repo}-${version}-${os}-${arch}.zip")
        message(
            VERBOSE
            "searching for the prebuilt slang-llvm library in ${version_url}"
        )
        check_assets_for_file("${json_content}" "${desired_zip}" file_found)

        if(file_found)
            set(${out_var} "${version}" PARENT_SCOPE)
            return()
        endif()
        message(
            WARNING
            "Failed to find ${desired_zip} in release assets for ${version} from ${version_url}\nFalling back to latest version if it differs"
        )
    else()
        set(w
            "Failed to download release info for version ${version} from ${version_url}\nFalling back to latest version if it differs"
        )
        if(status_code EQUAL 22)
            set(w
                "${w}\nIf you think this is failing because of GitHub API rate limiting, Github allows a higher limit if you use a token. Try the cmake option -DSLANG_GITHUB_TOKEN=your_token_here"
            )
        endif()

        message(WARNING ${w})
    endif()

    # If not found, get the latest release tag
    get_latest(${owner} ${repo} ${os} ${arch} "${github_token}" latest_version)
    if(NOT DEFINED latest_version)
        return()
    endif()
    set(${out_var} "${latest_version}" PARENT_SCOPE)
endfunction()

function(get_best_slang_binary_release_url github_token out_var)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64|AMD64")
        set(arch "x86_64")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|ARM64|arm64")
        set(arch "aarch64")
    else()
        message(
            WARNING
            "Unsupported architecture for slang binary releases: ${CMAKE_SYSTEM_PROCESSOR}"
        )
        return()
    endif()

    if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
        set(os "windows")
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
        set(os "macos")
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        set(os "linux")
    else()
        message(
            WARNING
            "Unsupported operating system for slang binary releases: ${CMAKE_SYSTEM_NAME}"
        )
        return()
    endif()

    set(owner "shader-slang")
    set(repo "slang")

    # This is the first version which distributed libslang-llvm.so, if it's
    # older than that then someone didn't fetch tags, emit a message and
    # fallback to the latest release
    if(${SLANG_VERSION_NUMERIC} VERSION_LESS "2024.1.27")
        if(${SLANG_VERSION_NUMERIC} VERSION_EQUAL "0.0.0")
            message(
                VERBOSE
                "The detected version of slang is ${SLANG_VERSION_NUMERIC}, fetching libslang-llvm from the latest release"
            )
        else()
            message(
                WARNING
                "The detected version of slang ${SLANG_VERSION_NUMERIC} is very old (probably you haven't fetched tags recently?), libslang-llvm will be fetched from the latest release rather than the one matching ${SLANG_VERSION_NUMERIC}"
            )
        endif()
        get_latest(
            ${owner}
            ${repo}
            ${os}
            ${arch}
            "${github_token}"
            release_version
        )
    else()
        check_release_and_get_latest(
            ${owner}
            ${repo}
            ${SLANG_VERSION_NUMERIC}
            ${os}
            ${arch}
            "${github_token}"
            release_version
        )
    endif()
    if(DEFINED release_version)
        message(
            VERBOSE
            "Found a version of libslang-llvm.so in ${release_version}"
        )
        set(${out_var}
            "https://github.com/${owner}/${repo}/releases/download/v${release_version}/slang-${release_version}-${os}-${arch}.zip"
            PARENT_SCOPE
        )
    endif()
endfunction()
