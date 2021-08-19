import argparse
import asyncio


from tools.linter.lint import Linter
from tools.linter.install import clang_format
from tools.linter.utils import CommandResult, run_cmd


class ClangFormat(Linter):
    name = "clang-format"
    exe = clang_format.INSTALLATION_PATH
    options = argparse.Namespace(
        paths=["c10", "torch/csrc/jit", "test/cpp/jit", "test/cpp/tensorexpr"],
        regex=[".*\\.(h|cpp|cc|c|hpp)$"],
        glob=[],
        error_if_changed=False,
        max_processes=50,
    )

    def build_parser(self, parser):
        parser.add_argument(
            "--error-if-changed",
            action="store_true",
            default=self.options.error_if_changed,
            help="Determine whether running clang-format would produce changes",
        )
        parser.add_argument(
            "--max-processes",
            type=int,
            default=self.options.max_processes,
            help="Maximum number of subprocesses to create to format files in parallel",
        )

    async def run_clang_format_on_file(
        self, filename: str, semaphore: asyncio.Semaphore, verbose: bool = False,
    ) -> CommandResult:
        """
        Run clang-format on the provided file.
        """
        # -style=file picks up the closest .clang-format, -i formats the files inplace.
        cmd = f"{self.exe} -style=file -i {filename}"
        async with semaphore:
            result = await run_cmd(cmd)
        if verbose:
            print(f"Formatted {filename}")
        return result

    async def file_clang_formatted_correctly(
        self, filename: str, semaphore: asyncio.Semaphore, verbose: bool = False,
    ) -> CommandResult:
        """
        Checks if a file is formatted correctly and returns True if so.
        """
        ok = True
        # -style=file picks up the closest .clang-format
        cmd = f"{self.exe} -style=file {filename}"

        async with semaphore:
            result = await run_cmd(cmd)

        formatted_contents = result.stdout

        # Compare the formatted file to the original file.
        with open(filename) as orig:
            orig_contents = orig.read().strip()
            if formatted_contents != orig_contents:
                result.returncode = -1

                # Reset stdout to avoid printing the contents of the file
                result.stdout = filename
                if verbose:
                    print(f"{filename} is not formatted correctly")

        if not result.failed():
            result.stdout = ""
        return result

    async def run(self, files, line_filters=None, options=options):
        result = CommandResult(0, "", "")
        semaphore = asyncio.Semaphore(options.max_processes)
        if options.error_if_changed:
            for task in asyncio.as_completed(
                [
                    self.file_clang_formatted_correctly(f, semaphore, options.verbose)
                    for f in files
                ]
            ):
                result += await task

            if not result.failed():
                result.stdout = "All files formatted correctly"
            else:
                result.returncode = -1
                result.stderr = "Some files not formatted correctly"
        else:
            results = await asyncio.gather(
                *[
                    self.run_clang_format_on_file(f, semaphore, options.verbose)
                    for f in files
                ]
            )

            result = sum(results, result)

        if result.failed():
            result.stdout += (
                "\nclang-format failures found! Run:\n"
                "   $ tools/linter/lint.py clang-format\n"
                "to fix this error.\n"
                "For more info, see: https://github.com/pytorch/pytorch/wiki/clang-format"
            )

        return result
