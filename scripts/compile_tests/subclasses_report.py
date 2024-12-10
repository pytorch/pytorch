from common import compute_pass_rate_aot_eager_subclasses
from download_reports import download_reports


if __name__ == "__main__":
    commit = "06b29c60afb2132b71c8b9a4ed077cefa52c75d1"
    # commit = "9fa04618c468233f8a1568c7f91126a2a3459cff"
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
