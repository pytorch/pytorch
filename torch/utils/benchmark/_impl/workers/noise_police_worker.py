import os
import typing

from torch.utils.benchmark._impl.workers import subprocess_worker


class NoisePoliceWorker(subprocess_worker.SubprocessWorker):

    @property
    def args(self) -> typing.List[str]:
        return [
            "sudo",
            "systemd-run",
            "--slice=workload.slice",
            "--same-dir",
            "--wait",
            "--collect",
            "--service-type=exec",
            "--pty",
            f'--uid={os.getlogin()}',
            f'--setenv=PATH="{os.environ["PATH"]}"',
        ] + super().args
