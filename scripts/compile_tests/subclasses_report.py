from common import compute_pass_rate_aot_eager_subclasses
from download_reports import download_reports


if __name__ == "__main__":
    # Result exist, but only 40 testcases
    # commit = "06b29c60afb2132b71c8b9a4ed077cefa52c75d1"
    # commit = "267aa6485765da9ed454779fcf518aed6a698632"
    # commit = "35172a50711beaf69ef0bcbafcafcbb3bc4c19f4"
    # commit = "0891fae132f2dd883f076a7819871e18fa3266a6"

    commit = "876b7d711f38bf9f2efcb6a57a58ffd396e5ad57"
    aot_eager311, subclasses311 = download_reports(
        commit, ("aot_eager311", "subclasses311")
    )
    print("compute pass rate py3.11 aot_eager vs subclasses")
    compute_pass_rate_aot_eager_subclasses(aot_eager311, subclasses311)

    # aot_eager39, subclasses39 = download_reports(
    #     commit, ("aot_eager39", "subclasses39")
    # )
    # print("compute pass rate py3.9 aot_eager vs subclasses")
    # compute_pass_rate(aot_eager39, subclasses39)
