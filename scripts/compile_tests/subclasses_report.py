from common import compute_pass_rate
from download_reports import download_reports

if __name__ == "__main__":
    commit = "8cb1794182edf8d27e1c68f6ce645f01a6402f09"
    aot_eager311, subclasses311 = download_reports(
        commit, ("aot_eager311", "subclasses311")
    )
    compute_pass_rate(aot_eager311, subclasses311)
