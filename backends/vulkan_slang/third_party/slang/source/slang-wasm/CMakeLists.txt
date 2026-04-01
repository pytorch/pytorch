#
# WebAssembly bindings for Slang
#
# This is an executable target because emcmake produces .a files without bindings if you just create a static library
# https://stackoverflow.com/questions/63622009/static-library-built-with-cmake-as-a-with-emscripten-instead-of-wasm-js
if(EMSCRIPTEN)
    slang_add_target(
        .
        EXECUTABLE
        EXCLUDE_FROM_ALL
        USE_FEWER_WARNINGS
        LINK_WITH_PRIVATE
            miniz
            lz4_static
            slang
            core
            compiler-core
            slang-capability-defs
            slang-capability-lookup
            slang-fiddle-output
            slang-lookup-tables
        INCLUDE_DIRECTORIES_PUBLIC ${slang_SOURCE_DIR}/include .
    )
    # To generate binding code
    target_link_options(
        slang-wasm
        PUBLIC "--bind" "--emit-tsd" "interface.d.ts" "-s" "EXPORT_ES6=1"
    )
endif()
