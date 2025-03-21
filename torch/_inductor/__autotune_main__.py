import argparse
import logging
import os

from torch._inductor.autotune_process import TuningProcess
from torch._inductor.compile_worker.utils import _async_compile_initializer


log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=int)
    parser.add_argument("--read-fd", type=int)
    parser.add_argument("--write-fd", type=int)
    args = parser.parse_args()
    read_pipe = os.fdopen(args.read_fd, "rb")
    write_pipe = os.fdopen(args.write_fd, "wb")

    try:
        # Ensures the subprocess exits if the parent crashes:
        _async_compile_initializer(args.parent)
        TuningProcess.process_main(read_pipe, write_pipe)
    except Exception:
        log.exception("Uncaught exception in autotune subprocess")
    finally:
        read_pipe.close()
        write_pipe.close()


if __name__ == "__main__":
    main()
