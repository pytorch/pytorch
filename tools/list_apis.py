import argparse
import inspect
import importlib
import sys

from types import ModuleType
from collections import namedtuple
from typing import List, Any, Set


CrawlError = namedtuple("CrawlError", ("reason", "path"))
CrawlResult = namedtuple("CrawlResult", ("public", "private", "errors", "module"))


def err_print(*args: Any) -> None:
    print(*args, file=sys.stderr)


def should_skip(obj: Any) -> bool:
    system_modules = [
        "os",
        "sys",
        "re",
        "shutil",
        "numpy",
    ]
    for name in system_modules:
        if obj == importlib.import_module(name):
            return True

    return False


def get_name(path: List[str], name: str) -> str:
    return ".".join(path + [name])


def add_item(path: List[str], name: str, out: CrawlResult) -> None:
    is_private = False
    for item in path:
        if item.startswith("_"):
            is_private = True
            break
    if name.startswith("_"):
        is_private = True

    if is_private:
        out.private.append(get_name(path, name))
    else:
        out.public.append(get_name(path, name))


def get_module_attributes(module: ModuleType) -> List[str]:
    attrs = []

    try:
        attrs += dir(module)
    except Exception as e:
        err_print(e)
        pass

    try:
        attrs += getattr(module, "__all__", [])
    except Exception as e:
        err_print(e)
        pass

    return attrs


def crawl_helper(
    obj: Any,
    name: str,
    path: List[str],
    seen_modules: Set[ModuleType],
    out: CrawlResult,
) -> None:
    if isinstance(obj, ModuleType):
        if obj in seen_modules:
            return
        seen_modules.add(obj)
        if should_skip(obj):
            return

        attrs = get_module_attributes(obj)

        attrs = sorted(list(set(attrs)))
        add_item(path, name, out)

        for attr in attrs:
            try:
                next_obj = getattr(obj, attr)
                crawl_helper(next_obj, attr, path + [name], seen_modules, out)
            except AttributeError as e:
                err_print(e)
                err_print(f"ERROR: unaccessible attribute {get_name(path, name)}")
                out.errors.append(
                    CrawlError(
                        reason="unaccessible attribute", path=get_name(path, name)
                    )
                )

    elif inspect.isclass(obj):
        attrs = dir(obj)

        for attr in attrs:
            add_item(path + [name], attr, out)
    else:
        add_item(path, name, out)


def crawl(module: ModuleType) -> CrawlResult:
    result = CrawlResult(
        module=module,
        public=[],
        private=[],
        errors=[],
    )
    seen_modules: Set[ModuleType] = set()
    crawl_helper(module, module.__name__, [], seen_modules, result)
    return result


def main(module: ModuleType, public: bool, private: bool, errors: bool) -> None:
    r = crawl(module)

    if public:
        for item in r.public:
            print(item)
    if private:
        for item in r.private:
            print(item)
    if errors:
        for item in r.errors:
            print(item)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Recursively list all reachable objects in a Python module (library)."
        + " This is used in PyTorch releases to determine the API surface changes"
    )
    parser.add_argument("--module", help="module to crawl", required=True)

    def add_flag(name: str, default: bool, help: str) -> None:
        parser.add_argument(f"--{name}", dest=name, help=help, action="store_true")
        parser.add_argument(f"--no-{name}", dest=name, help=help, action="store_false")
        parser.set_defaults(**{name: default})

    add_flag(name="public", default=True, help="list public APIs")
    add_flag(
        name="private",
        default=False,
        help="list private APIs (those that start with a _)",
    )
    add_flag(name="errors", default=False, help="show errors (unreachable APIs)")

    args = parser.parse_args()

    main(
        module=importlib.import_module(args.module),
        public=args.public,
        private=args.private,
        errors=args.errors,
    )
