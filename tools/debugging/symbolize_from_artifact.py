import argparse
import multiprocessing
from pathlib import Path
import os
import subprocess
import sys
import shutil


def eprint(*args):
    print(*args, file=sys.stderr)


def run_cmd(cmd, **kwargs):
    cmd = [str(x) for x in cmd]
    eprint("$", " ".join(cmd))
    return subprocess.run(cmd, **kwargs)


def create_symbols(dump_syms_exe, pytorch_build_dir, symbols_dir):
    """
    Creates symbols directory for minidump_stackwalk as described here:
    https://chromium.googlesource.com/breakpad/breakpad/+/master/docs/linux_starter_guide.md#producing-symbols-for-your-application

    The basic structure is parent -> library name -> module ID -> symbols text file,
    for example:

        symbols/
            - libtorch_cpu.so/
                - ABC123/
                    - libtorch_cpu.so.sym
    """
    # walk the bin dir, dump_syms_exe, move to appropriate directory
    # https://chromium.googlesource.com/chromium/chromium/+/34599b0bf7a14ab21a04483c46ecd9b5eaf86704/components/breakpad/tools/generate_breakpad_symbols.py
    # Walk the directory of torch libraries, generate symbols for each one
    for lib in (pytorch_build_dir / "lib").glob("*.so*"):
        r = run_cmd(
            [dump_syms_exe, lib.resolve()],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if r.returncode != 0:
            print("Error creating symbols for", lib)
            continue

        stdout = r.stdout.decode()
        first_line = stdout[0 : stdout.find("\n")].split()
        module_id = first_line[3]
        lib_name = first_line[4]

        module_dir = symbols_dir / lib_name / module_id
        os.makedirs(module_dir, exist_ok=True)

        with open(module_dir / f"{lib_name}.sym", "w") as f:
            f.write(stdout)


def stackwalk(stackwalk_exe, crash, symbols_dir):
    """
    Run minidump_stackwalk from breakpad and output the results
    """
    # run minidump stackwalk with the create_symbols and show output
    r = run_cmd(
        [stackwalk_exe, crash, symbols_dir],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    print(r.stdout.decode())


def build_breakpad(breakpad_dir):
    """
    Clone and build breakpad
    """
    run_cmd(
        ["git", "clone", "https://github.com/google/breakpad.git", ".breakpad"],
        cwd=breakpad_dir.parent,
        check=True,
    )
    run_cmd(
        [
            "git",
            "clone",
            "https://chromium.googlesource.com/linux-syscall-support",
            "src/third_party/lss",
        ],
        cwd=breakpad_dir,
        check=True,
    )
    run_cmd(["./configure"], cwd=breakpad_dir, check=True)
    run_cmd(["make", "-j", multiprocessing.cpu_count()], cwd=breakpad_dir, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--minidump", required=True, help="minidump .dmp file")
    parser.add_argument(
        "--pytorch-build-dir",
        required=True,
        help="pytorch/build directory (must have 'lib' folder)",
    )
    args = parser.parse_args()

    symbols_dir = Path("symbols")

    dump_syms_exe = shutil.which("dump_syms")
    minidump_stackwalk_exe = shutil.which("minidump_stackwalk")

    if dump_syms_exe is None or minidump_stackwalk_exe is None:
        # Binaries not found in path, check hardcoded build location
        breakpad_home = Path.home() / ".breakpad"
        if not breakpad_home.exists():
            print(
                "breakpad binaries not found in PATH, clone and build it from source? (y/n) ",
                end="",
            )
            c = input().lower()
            if c == "y":
                build_breakpad(breakpad_home)
            else:
                print("Can't generate symbols without breakpad, quitting")
                exit(1)

        dump_syms_exe = (
            breakpad_home / "src" / "tools" / "linux" / "dump_syms" / "dump_syms"
        )
        minidump_stackwalk_exe = (
            breakpad_home / "src" / "processor" / "minidump_stackwalk"
        )

    print(dump_syms_exe)
    print(minidump_stackwalk_exe)

    if not symbols_dir.exists():
        eprint(f"{symbols_dir} doesn't exist, generating symbols")
        os.makedirs(symbols_dir, exist_ok=True)

        create_symbols(
            dump_syms_exe,
            Path(args.pytorch_build_dir),
            symbols_dir,
        )

    stackwalk(
        minidump_stackwalk_exe,
        args.minidump,
        symbols_dir,
    )
