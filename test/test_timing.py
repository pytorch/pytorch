from torch.utils.benchmark import Language, Timer

timer = Timer(
    stmt="x + y",
    setup="""
        x = torch.ones(0)
        y = torch.ones(0)
    """
)

cpp_timer = Timer(
    "x + y;",
    """
        auto x = torch::ones({0});
        auto y = torch::ones({0});
    """,
    language=Language.CPP,
)

stats = timer.collect_callgrind()
print(stats)

cpp_stats = cpp_timer.collect_callgrind()
print(cpp_stats)
