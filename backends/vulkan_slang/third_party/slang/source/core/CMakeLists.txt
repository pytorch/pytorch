slang_add_target(
    .
    STATIC
    EXPORT_MACRO_PREFIX SLANG
    EXCLUDE_FROM_ALL
    USE_EXTRA_WARNINGS
    LINK_WITH_PRIVATE miniz lz4_static Threads::Threads ${CMAKE_DL_LIBS}
    LINK_WITH_PUBLIC unordered_dense::unordered_dense
    INCLUDE_DIRECTORIES_PUBLIC
        ${slang_SOURCE_DIR}/source
        ${slang_SOURCE_DIR}/include
)
