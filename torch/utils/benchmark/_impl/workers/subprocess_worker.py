import datetime
import io
import os
import marshal
import shutil
import subprocess
import sys
import pathlib
import tempfile
import textwrap
import time
import typing

from torch.utils.benchmark._impl.workers import base
from torch.utils.benchmark._impl.workers import subprocess_worker_handler


class SubprocessWorker(base.WorkerBase):
    """Open a subprocess using `python -i`, and use it to execute code.

    The launch command is determined by the `.args` property so that subclasses
    can override (generally suppliment) the process launch command. This class
    handles the complexity of communication and fault handling.

    TODO:
        1) Log inputs and outputs to `self._working_dir`
        2) Check AST before sending to subprocess
        3) Bubble failures up to parent process
    """

    _working_dir: str

    def __init__(self) -> None:
        super().__init__()

        self._stdin: str = os.path.join(self.working_dir, "stdin.log")
        self._stdout: str = os.path.join(self.working_dir, "stdout.txt")
        self._stderr: str = os.path.join(self.working_dir, "stderr.txt")

        self._cmd_begin = os.path.join(self.working_dir, "cmd_begin")
        self._cmd_success = os.path.join(self.working_dir, "cmd_success")
        self._cmd_failed = os.path.join(self.working_dir, "cmd_failed")
        self._cmd_finished = os.path.join(self.working_dir, "cmd_finished")

        pathlib.Path(self._stdin).touch()
        self._stdout_f = open(self._stdout, "wt")
        self._stderr_f = open(self._stderr, "wt")

        self._payloads = os.path.join(self.working_dir, "payloads")
        os.mkdir(self._payloads)

        self._proc = subprocess.Popen(
            args=self.args,
            stdin=subprocess.PIPE,
            stdout=io.TextIOWrapper(self._stdout_f, encoding="utf-8", write_through=True),
            stderr=io.TextIOWrapper(self._stderr_f, encoding="utf-8", write_through=True),
            encoding="utf-8",
            bufsize=1,
        )
        self.write_stdin(subprocess_worker_handler.HANDLER_SOURCE)

    def write_stdin(self, msg: str) -> None:
        if msg:
            if msg[0] != "\n":
                msg = f"\n{msg}"

            if msg[-1] != "\n":
                msg = f"{msg}\n"

        if self._proc.poll() is not None:
            raise ValueError("`self._proc` has exited. Cannot write to stdin.")

        # Log stdin for debugging. (With time added for convenience.)
        with open(self._stdin, "at", encoding="utf-8") as f:
            now = datetime.datetime.now().strftime("[%Y-%m-%d] %H:%M:%S.%f")
            f.write(f"# {now}\n{msg}\n")

        # Actually write to proc stdin.
        self._proc.stdin.flush()
        self._proc.stdin.write(msg)
        self._proc.stdin.flush()

    @property
    def working_dir(self) -> str:
        if getattr(self, "_working_dir", None) is None:
            self._working_dir = tempfile.mkdtemp()
        return self._working_dir

    @property
    def args(self) -> typing.List[str]:
        return [sys.executable, "-i", "-u"]

    @property
    def in_process(self) -> bool:
        return False

    def run(self, snippet: str) -> None:
        self._run(snippet)

    def store(self, name: str, value: typing.Any, in_memory: bool = False) -> None:
        if in_memory:
            raise NotImplementedError("SubprocessWorker does not support `in_memory`")

        # NB: we convert the bytes to a hex string to avoid encoding issues.
        self._run(textwrap.dedent(f"""
            import marshal
            {name} = marshal.loads(bytes.fromhex(
                {repr(marshal.dumps(value).hex())}
            ))
        """))

    def load(self, name: str) -> typing.Any:
        fname = datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S_%f.payload")
        fpath = os.path.join(self._payloads, fname)

        self._run(textwrap.dedent(f"""
            import marshal
            with open({repr(fpath)}, "wb") as f:
                marshal.dump({name}, f)
        """))

        with open(fpath, "rb") as f:
            return marshal.load(f)

    def _run(self, snippet: str) -> None:
        """Helper method for running code in a subprocess."""

        # Get initial state for stdout and stderr so we can print new lines
        # if the command fails.
        stdout_stat = os.stat(self._stdout)
        stderr_stat = os.stat(self._stderr)

        for fname in (
            self._cmd_begin,
            self._cmd_success,
            self._cmd_failed,
            self._cmd_finished
        ):
            if os.path.exists(fname):
                os.remove(fname)

        self.write_stdin(textwrap.dedent(f"""
            _subprocess_snippet_handler(
                snippet={repr(snippet)},
                begin_fpath={repr(self._cmd_begin)},
                success_fpath={repr(self._cmd_success)},
                failure_fpath={repr(self._cmd_failed)},
                finished_fpath={repr(self._cmd_finished)},
            )
        """).strip())

        # TODO:
        #   Event loop
        #   Finer grained error handling. (e.g. timeout)
        #   Plumb exceptions to parent.
        while True:
            if os.path.exists(self._cmd_finished):
                break

            time.sleep(0.001)

        # It's a pretty poor user experience for errors to just vanish into
        # the void, so we grab all new additions to stdout and stderr and
        # include them in the error message.
        if os.path.exists(self._cmd_failed):
            with open(self._stdout, "rb") as f:
                _ = f.seek(stdout_stat.st_size)
                stdout = f.read().decode("utf-8").strip()

            with open(self._stderr, "rb") as f:
                _ = f.seek(stderr_stat.st_size)
                stderr = f.read().decode("utf-8").strip()

            raise ValueError(
                "Cmd failed.\n"
                f"    stdout:\n{textwrap.indent(stdout, ' ' * 8)}\n\n"
                f"    stderr:\n{textwrap.indent(stderr, ' ' * 8)}"
            )

    def __del__(self):
        if self._proc.poll() is None:
            try:
                self._proc.terminate()

            except PermissionError:
                # NoisePoliceWorker runs under sudo, and thus will not allow
                # SIGTERM to be sent. Note that because __del__ is sometimes
                # called during program shutdown, we can't use
                # `self.write_stdin`. (Because some of the modules will have
                # already been unloaded.)
                self._proc.stdin.write("exit()\n")
                self._proc.stdin.flush()

        self._stdout_f.close()
        self._stderr_f.close()

        shutil.rmtree(self._working_dir, ignore_errors=True)
