import json
import sys

import torch._logging.scribe as scribe


def main():
    duration = int(sys.argv[1])
    scribe.open_source_signpost(
        subsystem="pr_time_benchmarks",
        name="duration",
        parameters=json.dumps(duration),
    )


if __name__ == "__main__":
    main()
