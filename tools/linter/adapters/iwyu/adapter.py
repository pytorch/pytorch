import argparse
import concurrent.futures
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Any
import dataclasses

from .lint_message import LintSeverity, LintMessage
from .generate_diff import process_iwyu_output, FixIncludeFlags


IS_WINDOWS: bool = os.name == "nt"


def as_posix(name: str) -> str:
    return name.replace("\\", "/") if IS_WINDOWS else name


def _run_command(
    args: List[str], **kwargs: Any
) -> "subprocess.CompletedProcess[bytes]":
    logging.debug("$ %s", " ".join(args))
    start_time = time.monotonic()
    try:
        return subprocess.run(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=IS_WINDOWS,  # So batch scripts are found.
            **kwargs,
        )
    finally:
        end_time = time.monotonic()
        logging.debug("took %dms", (end_time - start_time) * 1000)


def check_file(
    filename: str, binary: str, build_dir: Path, mapping_files: List[Path]
) -> List[LintMessage]:
    mapping_file_args = []
    for path in mapping_files:
        mapping_file_args += ["-Xiwyu", f"--mapping_file={path}"]

    try:
        with open(filename, "rb") as f:
            original = f.read()

        iwyu_output = _run_command(
            [
                binary,
                f"-p={build_dir}",
                filename,
                "--",
                "-Wno-unknown-warning-option",
                "-Xiwyu",
                "--no_fwd_decls",
                *mapping_file_args,
            ]
        )

        return process_iwyu_output(
            input=iwyu_output.stdout.decode("utf-8"),
            flags=FixIncludeFlags(
                reorder=True,
                comments=False,
                basedir=str(build_dir),
                # Separate includes by "top level directory"
                separate_project_includes="<tld>",
            ),
        )
    except (OSError, subprocess.CalledProcessError) as err:
        return [
            LintMessage(
                path=filename,
                line=None,
                char=None,
                code="IWYU",
                severity=LintSeverity.ADVICE,
                name="command-failed",
                original=None,
                replacement=None,
                description=(
                    f"Failed due to {err.__class__.__name__}:\n{err}"
                    if not isinstance(err, subprocess.CalledProcessError)
                    else (
                        "COMMAND (exit code {returncode})\n"
                        "{command}\n\n"
                        "STDERR\n{stderr}\n\n"
                        "STDOUT\n{stdout}"
                    ).format(
                        returncode=err.returncode,
                        command=" ".join(as_posix(x) for x in err.cmd),
                        stderr=err.stderr.decode("utf-8").strip() or "(empty)",
                        stdout=err.stdout.decode("utf-8").strip() or "(empty)",
                    )
                ),
            )
        ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Format files with clang-format.",
        fromfile_prefix_chars="@",
    )
    parser.add_argument(
        "--binary",
        required=True,
        help="clang-format binary path",
    )
    parser.add_argument(
        "--build_dir",
        required=True,
        help=(
            "Where the compile_commands.json file is located. "
            "Gets passed to iwyu_tool -p"
        ),
    )
    parser.add_argument(
        "--mapping_file",
        nargs="*",
        required=True,
        help="Path to include-what-you-use mapping files",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="verbose logging",
    )
    parser.add_argument(
        "filenames",
        nargs="+",
        help="paths to lint",
    )
    args = parser.parse_args()

    logging.basicConfig(
        format="<%(threadName)s:%(levelname)s> %(message)s",
        level=logging.NOTSET
        if args.verbose
        else logging.DEBUG
        if len(args.filenames) < 1000
        else logging.INFO,
        stream=sys.stderr,
    )

    binary = os.path.normpath(args.binary) if IS_WINDOWS else args.binary
    if not Path(binary).exists():
        lint_message = LintMessage(
            path=None,
            line=None,
            char=None,
            code="IWYU",
            severity=LintSeverity.ERROR,
            name="init-error",
            original=None,
            replacement=None,
            description=(
                f"Could not find iwyu binary at {binary}, "
                "did you forget to run `lintrunner init`?"
            ),
        )
        print(json.dumps(dataclasses.asdict(lint_message)), flush=True)
        sys.exit(0)

    abs_build_dir = Path(args.build_dir).resolve()
    abs_mapping_files = [Path(p).resolve() for p in args.mapping_file]

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=os.cpu_count(),
        thread_name_prefix="Thread",
    ) as executor:
        futures = {
            executor.submit(check_file, x, binary, abs_build_dir, abs_mapping_files): x
            for x in args.filenames
        }
        for future in concurrent.futures.as_completed(futures):
            try:
                for lint_message in future.result():
                    print(json.dumps(dataclasses.asdict(lint_message)), flush=True)
            except Exception:
                logging.critical('Failed at "%s".', futures[future])
                raise


if __name__ == "__main__":
    main()
