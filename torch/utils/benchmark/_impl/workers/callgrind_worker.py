import os
import re
import typing

from torch.utils.benchmark._impl.workers import subprocess_worker


class CallgrindWorker(subprocess_worker.SubprocessWorker):
    def __init__(self):
        self._callgrind_out_dir = os.path.join(self.working_dir, "callgrind")
        os.mkdir(self._callgrind_out_dir)
        self._out_prefix = os.path.join(self._callgrind_out_dir, "callgrind.out")
        super().__init__()

    @property
    def out_files(self) -> typing.List[str]:
        pattern = re.compile("callgrind.out.([0-9]+)$")
        files = [
            os.path.join(self._callgrind_out_dir, i)
            for i in os.listdir(self._callgrind_out_dir)
            if pattern.match(i)
        ]
        return sorted(files, key=lambda x: int(pattern.search(x).groups()[0]))

    @property
    def args(self) -> typing.List[str]:
        return [
            "valgrind",
            "--tool=callgrind",
            f"--callgrind-out-file={self._out_prefix}",
            "--dump-line=yes",
            "--dump-instr=yes",
            "--instr-atstart=yes",
            "--collect-atstart=no",
        ] + super().args
