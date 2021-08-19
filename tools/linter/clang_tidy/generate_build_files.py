import sys
import os
import asyncio

from tools.linter.utils import CommandResult, run_cmd


async def update_submodules() -> CommandResult:
    return await run_cmd(["git", "submodule", "update", "--init", "--recursive"])


async def gen_compile_commands() -> CommandResult:
    os.environ["USE_NCCL"] = "0"
    os.environ["USE_DEPLOY"] = "1"
    os.environ["CC"] = "clang"
    os.environ["CXX"] = "clang++"
    return await run_cmd(["time", sys.executable, "setup.py", "--cmake-only", "build"])


async def run_autogen() -> CommandResult:
    result = await run_cmd(
        [
            "time",
            sys.executable,
            "-m",
            "tools.codegen.gen",
            "-s",
            "aten/src/ATen",
            "-d",
            "build/aten/src/ATen",
        ]
    )

    result += await run_cmd(
        [
            "time",
            sys.executable,
            "tools/setup_helpers/generate_code.py",
            "--declarations-path",
            "build/aten/src/ATen/Declarations.yaml",
            "--native-functions-path",
            "aten/src/ATen/native/native_functions.yaml",
            "--nn-path",
            "aten/src",
        ]
    )

    return result


async def generate_build_files() -> CommandResult:
    return (
        await update_submodules()
        + await gen_compile_commands()
        + await run_autogen()
    )


if __name__ == "__main__":
    out = asyncio.get_event_loop().run_until_complete(generate_build_files())
    print(out)

