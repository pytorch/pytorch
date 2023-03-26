def define_targets(rules):
    rules.cc_library(
        name = "shadow_storage",
        srcs = ["shadow_storage.cpp"],
        hdrs = ["shadow_storage.h"],
        deps = [
            "//c10/macros",
            "//c10/util:base",
        ],
        visibility = [
            "//c10/core:__pkg__",
            "//c10/test/core/impl/cow:__pkg__",
        ],
    )

    rules.cc_library(
        name = "spy",
        srcs = ["spy.cpp"],
        hdrs = ["spy.h"],
        deps = [
            ":shadow_storage",
            "//c10/core:base",
            "//c10/macros",
            "//c10/util:base",
        ],
        visibility = ["//:__pkg__"],
    )

    rules.cc_library(
        name = "state_machine",
        srcs = ["state_machine.cpp"],
        hdrs = ["state_machine.h"],
        deps = [
            ":shadow_storage",
            "//c10/util:base",
        ],
        visibility = [
            "//c10/core:__pkg__",
            "//c10/test/core/impl/cow:__pkg__",
        ],
    )
