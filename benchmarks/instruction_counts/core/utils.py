import atexit
import shutil
import tempfile
from typing import List, Optional, Tuple

from core.api import AutogradMode, AutoLabels, RuntimeMode, TimerArgs, GroupedBenchmark
from core.jit import generate_torchscript_file
from core.types import Definition, FlatDefinition, FlatIntermediateDefinition, Label


_TEMPDIR: Optional[str] = None
def get_temp_dir() -> str:
    global _TEMPDIR
    if _TEMPDIR is None:
        temp_dir = tempfile.mkdtemp()
        atexit.register(shutil.rmtree, path=temp_dir)
        _TEMPDIR = temp_dir
    return _TEMPDIR


def _flatten(
    key_prefix: Label,
    sub_schema: Definition,
    result: FlatIntermediateDefinition
) -> None:
    for k, value in sub_schema.items():
        if isinstance(k, tuple):
            assert all(isinstance(ki, str) for ki in k)
            key_suffix: Label = k
        elif k is None:
            key_suffix = ()
        else:
            assert isinstance(k, str)
            key_suffix = (k,)

        key: Label = key_prefix + key_suffix
        if isinstance(value, (TimerArgs, GroupedBenchmark)):
            assert key not in result, f"duplicate key: {key}"
            result[key] = value
        else:
            assert isinstance(value, dict)
            _flatten(key_prefix=key, sub_schema=value, result=result)


def flatten(schema: Definition) -> FlatIntermediateDefinition:
    result: FlatIntermediateDefinition = {}
    _flatten(key_prefix=(), sub_schema=schema, result=result)

    # Ensure that we produced a valid flat definition.
    for k, v in result.items():
        assert isinstance(k, tuple)
        assert all(isinstance(ki, str) for ki in k)
        assert isinstance(v, (TimerArgs, GroupedBenchmark))
    return result


def unpack(definitions: FlatIntermediateDefinition) -> FlatDefinition:
    results: List[Tuple[Label, AutoLabels, TimerArgs]] = []

    for label, args in definitions.items():
        if isinstance(args, TimerArgs):
            auto_labels = AutoLabels(
                RuntimeMode.EXPLICIT,
                AutogradMode.EXPLICIT,
                args.language
            )
            results.append((label, auto_labels, args))

        else:
            assert isinstance(args, GroupedBenchmark)

            model_path: Optional[str] = None
            ts_model_setup = args.ts_model_setup
            if ts_model_setup is not None:
                name: str = re.sub(r'[^a-z0-9_]', '_', '_'.join(label).lower())
                name = f"{name}_{uuid.uuid4()}"
                model_path = generate_torchscript_file(ts_model_setup, name=name, temp_dir=get_temp_dir())

            for auto_labels, timer_args in args.flatten(model_path):
                results.append((label, auto_labels, timer_args))

    return tuple(results)
