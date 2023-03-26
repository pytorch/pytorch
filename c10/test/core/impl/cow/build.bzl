def define_targets(rules):
    rules.cc_test(
        name = "test",
        size = "small",
        srcs = ["test.cpp"],
        deps = [
            "@com_google_googletest//:gtest_main",
            "//c10/core/impl/cow:shadow_storage",
            "//c10/core/impl/cow:state_machine",
        ],
    )
