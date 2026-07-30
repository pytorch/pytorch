import argparse
from pathlib import Path
import subprocess
import json
import re
import shlex

# only Register*.cpp files that currently fail but are accepted temporarily
ALLOWLISTED_GENERATED_FILES: set[str] = {
    "RegisterMeta_0.cpp",  # see https://github.com/pytorch/pytorch/issues/191631
}

# strip ANSI colour codes for parsing
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

# extract diagnostics and file/line/message data
DIAGNOSTIC_RE = re.compile(
    r"^(?P<file>.*?):(?P<line>\d+):(?P<char>\d+):\s+"
    r"(?P<severity>warning|error):\s+"
    r"(?P<message>.*unused parameter.*)$",
    re.MULTILINE,
)

def compile_command_file(entry: dict[str: object]) -> Path:
    path = Path(str(entry["file"]))
    if not path.is_absolute():
        path = Path(str(entry["directory"])) / path
    return path.resolve()


# turn compile command into a lint-only compiler check
def rewrite_compile_command(args):
    rewritten = []
    skip_next = False

    for arg in args:
        if skip_next:
            skip_next = False
            continue

        if arg in {
            "-o", "-MF", "-MT", "-MQ"
        }:
            skip_next = True
            continue

        if arg in {
            "-c",
            "-Wno-unused-parameter",
            "-Wno-error=unused-parameter",
            "-MD",
            "-MMD",
        }:
            continue

        rewritten.append(arg)

    rewritten.extend([
        "-fsyntax-only",
        "-Wunused-parameter",
        "-Werror=unused-parameter",
        "-fdiagnostics-color=never",
    ])
    return rewritten


def emit_lint(path: Path | None, line: int | None, char: int | None, name: str, description: str):
    print(json.dumps({
        "path": str(path) if path is not None else None,
        "line":  line,
        "char":  char,
        "code":  "TORCHGEN_REGISTER_UNUSED_PARAMETER",
        "severity":  "error",
        "name":  name,
        "original":  None,
        "replacement":  None,
        "description":  str(description),
    }), flush=True)


def torchgen_register_unused_parameter_linter():
    # This lint is intentionally scoped to generated ATen registration wrappers.
    # The structured set_output_raw_strided code path that regressed is emitted
    # into Register*.cpp files.

    # Parse args from .lintrunner.toml:
    parser = argparse.ArgumentParser(fromfile_prefix_chars="@")
    parser.add_argument("--build-dir", required=True)
    parser.add_argument("filenames", nargs="*")

    args = parser.parse_args()

    _ = args.filenames  # only used by lintrunner to trigger this adapter
    build_dir = Path(args.build_dir)
    generated_dir = (build_dir / "aten" / "src" / "ATen").resolve()

    compile_commands_path = build_dir / "compile_commands.json"

    if not compile_commands_path.exists():
        emit_lint(
            path=None,
            line=None,
            char=None,
            name = "command-failed",
            description="Missing build/compile_commands.json. Generate build files before running this command",
        )
        return

    commands = json.loads(compile_commands_path.read_text())

    generated_files = []
    for entry in commands:
        path = compile_command_file(entry)
        if (
            path.parent == generated_dir
            and path.name.startswith("Register")
            and path.suffix == ".cpp"
            and "set_output_raw_strided" in path.read_text(errors="ignore")
        ):
            if path.name in ALLOWLISTED_GENERATED_FILES:
                continue
            generated_files.append((path, entry))
    generated_file_paths = {path for path, _ in generated_files}

    if not generated_files:
        emit_lint(
                path=None,
                line=None,
                char=None,
                name = "command-failed",
                description="No generated Register*.cpp files found in compile_commands.json.",
            )
        return

    for filename, entry in generated_files:
        if "arguments" in entry:
            args = list(entry["arguments"])
        else:
            args = shlex.split(entry["command"])

        rewritten_args = rewrite_compile_command(args)

        try:
            proc = subprocess.run(
                rewritten_args,
                cwd=entry.get("directory"),
                text=True,
                capture_output=True,
                check=False
            )
        except OSError as err:
            emit_lint(path=filename,
                      line=None,
                      char=None,
                      name="command-failed",
                      description=f"Failed to compile {filename}: {err}",
                      )
            continue

        output = proc.stdout + proc.stderr

        output = ANSI_RE.sub("", output)

        for match in DIAGNOSTIC_RE.finditer(output):
            diagnostic_path = Path(match["file"]).resolve()

            if diagnostic_path not in generated_file_paths:
                continue
            emit_lint(
                path=diagnostic_path,
                line=int(match["line"]),
                char=int(match["char"]),
                name="unused-parameter",
                description=match["message"],
            )

if __name__ == "__main__":
    torchgen_register_unused_parameter_linter()