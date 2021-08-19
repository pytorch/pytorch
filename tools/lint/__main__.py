import yaml
import click
import asyncio

from . import actions_local_runner
from . import utils

from typing import List, Dict, Any, Callable, Optional

GITIGNORE = actions_local_runner.REPO_ROOT / ".gitignore"

Kwargs = Dict[str, Any]


async def run_impl(linters: List[actions_local_runner.Check], **kwargs: Kwargs) -> int:
    if kwargs["verbose"]:
        utils.VERBOSE = True

    utils.log("Running", linters)
    utils.log("Running", kwargs)

    if kwargs.get("changed_only", None) and kwargs.get("path", []):
        raise RuntimeError("--changed-only and --path cannot be used together")

    files = None
    if kwargs.get("changed_only", None):
        files = actions_local_runner.changed_files()

    elif kwargs.get("path", []):
        files = []
        for path in kwargs["path"]:
            r = await actions_local_runner.shell_cmd(
                [
                    "git",
                    "ls-files",
                    "--cached",
                    "--others",
                    f"--exclude-from={GITIGNORE}",
                    path,
                ]
            )
            files += r.stdout.split("\n")
        # print(files)
        # exit(0)

    utils.log("Using files", files)
    coros = [linter.run(files) for linter in linters]
    results = await utils.gather(coros)

    for result in results:
        if isinstance(result, BaseException):
            raise result

    return 0 if all(results) else 1


def run(linters: List[actions_local_runner.Check], **kwargs: Kwargs) -> int:
    return asyncio.get_event_loop().run_until_complete(run_impl(linters, **kwargs))


common_options = {
    "path": click.option(
        "--path", help="Only lint files matching these globs", multiple=True
    ),
    "changed-only": click.option(
        "--changed-only",
        is_flag=True,
        help="Only lint files that have changed from origin/master",
    ),
    "diff-file": click.option(
        "--diff-file", help="Manually provide a diff to use for comparison"
    ),
    "verbose": click.option(
        "-v", "--verbose", help="Extra output for debugging", is_flag=True
    ),
}


def add_common_options(options: Optional[List[str]] = None) -> Callable[..., Any]:
    if options is None:
        options = ["path", "changed-only", "diff-file", "verbose"]

    def wrapper(fn: Callable[..., Any]) -> Any:
        for arg_name in options:
            fn = common_options[arg_name](fn)
        return fn

    return wrapper


@click.group()
def cli() -> None:
    """
    Lint PyTorch
    """
    pass


@cli.resultcallback()
def done(result: int, **kwargs: Kwargs) -> None:
    utils.log("Finished", result, kwargs)
    exit(result)


@cli.command(name="all")
@add_common_options()
@click.option("--generate-stubs", help="(mypy) Generate stubs", is_flag=True)
def _all(generate_stubs: bool, **kwargs: Kwargs) -> int:
    return run(
        [actions_local_runner.Flake8(), actions_local_runner.Mypy(generate_stubs)], **kwargs
    )


@cli.command()
@add_common_options()
def flake8(**kwargs: Kwargs) -> int:
    return run([actions_local_runner.Flake8()], **kwargs)


@cli.command()
@add_common_options()
@click.option("--generate-stubs", help="Generate stubs", is_flag=True)
def mypy(generate_stubs: bool, **kwargs: Kwargs) -> int:
    return run([actions_local_runner.Mypy(generate_stubs)], **kwargs)


@cli.command()
@add_common_options(["verbose"])
def quick_checks(**kwargs: Kwargs) -> int:
    with open(actions_local_runner.LINT_YML) as f:
        action = yaml.safe_load(f)
    action = action["jobs"]["quick-checks"]

    def _check(name: str) -> actions_local_runner.YamlStep:
        step = actions_local_runner.extract_step(name, action)
        return actions_local_runner.YamlStep(step, "quick-checks")

    checks = [
        _check("Ensure no trailing spaces"),
        _check("Ensure no tabs"),
        _check("Ensure no non-breaking spaces"),
        _check("Ensure canonical include"),
        _check("Ensure no versionless Python shebangs"),
        _check("Ensure no unqualified noqa"),
        _check("Ensure no unqualified type ignore"),
        _check("Ensure no direct cub include"),
        _check("Ensure correct trailing newlines"),
        _check("Ensure no raw cuda api calls"),
    ]

    return run(checks, **kwargs)


if __name__ == "__main__":
    cli()
