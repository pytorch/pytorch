from common import compute_pass_rate_aot_eager_subclasses
from download_reports import download_reports


if __name__ == "__main__":
    # commit = "f1e85e88d378253532a80a506dda98415b985dc0"
    # commit = "49fd00d0a472bf9524398dffb1f14e840b0f4b4b"
    # commit = "41efe136c41b5c22db57fb4d8d81ffaafb9bbc02"
    commit = "5c140e44fb1ae53e025f0388781c729a6d0294aa"
    eager313, dw313, aot_eager313, subclasses313 = download_reports(
        commit, ("eager313", "dynamo_wrapped313", "aot_eager313", "subclasses313")
    )
    print("compute pass rate py3.11 aot_eager vs subclasses")
    compute_pass_rate_aot_eager_subclasses(
        eager313, dw313, aot_eager313, subclasses313, name=commit[:8]
    )
