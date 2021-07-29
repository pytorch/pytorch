import argparse
import asyncio


from tools.linter.lint import Linter
from tools.linter.install import clang_format
from tools.linter.utils import CommandResult


class ClangFormat(Linter):
    name = "clang-format"
    exe = clang_format.INSTALLATION_PATH
    options = {
        **Linter.options,
        "paths": ["c10", "torch/csrc/jit", "test/cpp/jit", "test/cpp/tensorexpr"],
        "regex": [".*\\.(h|cpp|cc|c|hpp)$"],
    }

    def build_parser(self, parser):
        parser.add_argument(
            "-d",
            "--diff",
            action="store_true",
            default=False,
            help="Determine whether running clang-format would produce changes",
        )
        parser.add_argument("--verbose", "-v", action="store_true", default=False)
        parser.add_argument(
            "--max-processes",
            type=int,
            default=50,
            help="Maximum number of subprocesses to create to format files in parallel",
        )

    # TODO migrate to use CommandResult
    async def run_clang_format_on_file(
        self, filename: str, semaphore: asyncio.Semaphore, verbose: bool = False,
    ) -> None:
        """
        Run clang-format on the provided file.
        """
        # -style=file picks up the closest .clang-format, -i formats the files inplace.
        cmd = "{} -style=file -i {}".format(self.exe, filename)
        async with semaphore:
            proc = await asyncio.create_subprocess_shell(cmd)
            _ = await proc.wait()
        if verbose:
            print("Formatted {}".format(filename))

    # TODO migrate to use CommandResult
    async def file_clang_formatted_correctly(
        self, filename: str, semaphore: asyncio.Semaphore, verbose: bool = False,
    ) -> bool:
        """
        Checks if a file is formatted correctly and returns True if so.
        """
        ok = True
        # -style=file picks up the closest .clang-format
        cmd = "{} -style=file {}".format(self.exe, filename)

        async with semaphore:
            proc = await asyncio.create_subprocess_shell(
                cmd, stdout=asyncio.subprocess.PIPE
            )
            # Read back the formatted file.
            stdout, _ = await proc.communicate()

        formatted_contents = stdout.decode()
        # Compare the formatted file to the original file.
        with open(filename) as orig:
            orig_contents = orig.read()
            if formatted_contents != orig_contents:
                ok = False
                if verbose:
                    print("{} is not formatted correctly".format(filename))

        return ok

    def run(self, files, options):
        result = CommandResult(0, "", "")

        semaphore = asyncio.Semaphore(options.max_processes)
        if options.diff:
            for f in asyncio.as_completed(
                [
                    self.file_clang_formatted_correctly(f, semaphore, options.verbose)
                    for f in files
                ]
            ):
                ok &= await f

            if ok:
                print("All files formatted correctly")
            else:
                print("Some files not formatted correctly")
        else:
            await asyncio.gather(
                *[
                    self.run_clang_format_on_file(f, semaphore, options.verbose)
                    for f in files
                ]
            )

        return result
