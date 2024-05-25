def define_targets(rules):
    rules.cc_binary(
        name = "intrusive_ptr",
        srcs = ["intrusive_ptr_benchmark.cpp"],
        tags = ["benchmark"],
        deps = [
            "//c10/util:base",
            "@google_benchmark//:benchmark",
        ],
    )
