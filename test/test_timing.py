from torch.utils.benchmark import Timer

timer = Timer(
    stmt="x + y",
    setup="""
        x = torch.ones((16,))
        y = torch.ones((16,))
    """
)

stats = timer.collect_callgrind()
print(stats)
