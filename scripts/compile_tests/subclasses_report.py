from common import compute_pass_rate_aot_eager_subclasses
from download_reports import download_reports


if __name__ == "__main__":
    commit = "f1e85e88d378253532a80a506dda98415b985dc0"
    eager313, dw313, aot_eager313, subclasses313 = download_reports(
        commit, ("eager313", "dynamo_wrapped313", "aot_eager313", "subclasses313")
    )
    print("compute pass rate py3.11 aot_eager vs subclasses")
    compute_pass_rate_aot_eager_subclasses(eager313, dw313, aot_eager313, subclasses313)
