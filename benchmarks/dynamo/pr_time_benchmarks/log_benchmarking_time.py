import json
import sys

import torch._logging.scribe as scribe


def main():
    # Try to convert the command-line argument to an integer
    try:
        duration = int(sys.argv[1])
        scribe.open_source_signpost(
            subsystem="pr_time_benchmarks",
            name="duration",
            parameters=json.dumps(duration),
        )
    except ValueError:
        print("Error: Invalid input. Expecting integer")


if __name__ == "__main__":
    main()
