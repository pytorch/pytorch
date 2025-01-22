import dataclasses
from typing import Any, Optional, Union

import torch
from torch._dynamo.exc import UserError, UserErrorType
from torch.export.dynamic_shapes import (
    _check_dynamic_shapes,
    _DerivedDim,
    _Dim,
    _DimHint,
    _tree_map_with_path,
    Dim,
)
from torch.utils._pytree import tree_map

from .serialize import _dataclass_to_dict


@dataclasses.dataclass
class RootDim:
    """
    This represents a _Dim object.
    """

    min: int
    max: Union[int, None]
    derived: list[str]


@dataclasses.dataclass
class DynamicShapesSpec:
    """
    This stores a dynamic_shapes spec for de/serialization.
    """

    dynamic_shapes: Union[dict[str, Any], tuple[Any], list[Any], None]
    dims: dict[str, RootDim]


def _postprocess_serialized_shapes(
    dynamic_shapes: Union[dict[str, Any], tuple[Any], list[Any], None],
    dims: dict[str, dict[str, Union[int, list[str], None]]],
    to_dict: Optional[bool] = False,
) -> Union[DynamicShapesSpec, dict[str, Any]]:
    """
    Sorts dims and dumps to dictionary format.
    """
    from torch.utils._sympy.numbers import int_oo

    dims = {
        k: RootDim(
            min=v["min"],  # type: ignore[arg-type]
            max=None if v["max"] is int_oo else v["max"],  # type: ignore[arg-type]
            derived=sorted(v["derived"]),  # type: ignore[arg-type]
        )
        for k, v in sorted(dims.items())
    }
    spec = DynamicShapesSpec(dynamic_shapes=dynamic_shapes, dims=dims)
    if to_dict:
        return _dataclass_to_dict(spec)
    else:
        return spec


def _dump_dynamic_shapes(
    dynamic_shapes: Union[dict[str, Any], tuple[Any], list[Any], None],
    args: tuple[Any],
    kwargs: Optional[dict[str, Any]] = None,
    to_dict: Optional[bool] = False,
) -> Union[DynamicShapesSpec, dict[str, Any]]:
    """
    Utility function for dynamic shapes serialization, serializing a dynamic_shapes spec.
    Returns a DynamicShapesSpec dataclass containing 2 fields, "dynamic_shapes" and "dims".
    Uses args & kwargs to distinguish between tensor-level and dim-level specs (only for Nones).

    dynamic_shapes: A pytree structure mirroring the dynamic_shapes input to export():
        - Each tensor input is represented with a list of values, non-tensor inputs with None.
        - dynamic dimensions (i.e. symbols) in tensors and Dim enums are represented with strings.
        - static dimensions are represented with ints.

    dims: A dictionary mapping each symbol name to the min/max range and derived dim names.

    For example:
    ```
    dx = Dim("dx", min=4, max=16)
    dy = dx + 1

    inputs = (
        [
            torch.randn(4, 4),
            torch.randn(5, 4),
        ],
        torch.randn(4),
        torch.randn(4, 4),
        "hello",
    )
    dynamic_shapes = {
        "a": [
            (dx, 4),
            (dy, 4),
        ],
        "b": (Dim.STATIC,),
        "c": None,
        "d": None,
    }
    out = _dump_dynamic_shapes(dynamic_shapes, inputs, to_dict=True)
    ```
    would generate the following output:
    ```
    {
        'dynamic_shapes': (
            [
                ['dx', 4],
                ['dx + 1', 4],
            ],
            ['_DimHint.STATIC'],
            ['_DimHint.STATIC', '_DimHint.STATIC'],
            None,
        ),
        'dims': {
            'dx': {
                'min': 4,
                'max': 16,
                'derived': ['dx + 1'],
            },
        },
    }
    ```
    """
    dims: dict[str, dict[str, Any]] = {}

    def _standardize_shapes(path, tensor, shape):  # type: ignore[no-untyped-def]
        """
        Helps standardize the dynamic_shapes tree structure we serialize,
        returning lists for each tensor shape, handling tensor-level Nones.
        """
        if not isinstance(tensor, torch.Tensor):
            return None
        if shape is None:
            return [Dim.STATIC] * len(tensor.shape)  # type: ignore[attr-defined]

        out = []
        if isinstance(shape, dict):
            for i, s in enumerate(tensor.shape):
                out.append(s if shape.get(i) is None else shape.get(i))
        else:
            assert isinstance(shape, (tuple, list))
            for i, s in enumerate(tensor.shape):
                out.append(s if shape[i] is None else shape[i])
        return out

    def _track_dim_from_dims(
        val: Union[None, int, _DimHint, _Dim]
    ) -> Union[None, int, str]:
        """
        Tracks dims, ranges, derived dims from the standardized dynamic_shapes spec.
        """
        if val is None or isinstance(val, int):  # non-tensor input or static
            return val
        if isinstance(val, _DimHint):  # store enum as string
            return val.__class__.__name__ + "." + val.name

        assert isinstance(val, _Dim)

        # track root dim
        root = val.root if isinstance(val, _DerivedDim) else val  # type: ignore[attr-defined]
        if root.__name__ not in dims:
            dims[root.__name__] = {
                "min": root.min,  # type: ignore[attr-defined,union-attr]
                "max": root.max,  # type: ignore[attr-defined,union-attr]
                "derived": set(),
            }

        # track derived dims
        if isinstance(val, _DerivedDim):
            dims[root.__name__]["derived"].add(val.__name__)

        return val.__name__

    if dynamic_shapes is None:
        return {"dynamic_shapes": None, "dims": {}}

    # convert to tuple of specs, for each arg/kwarg
    kwargs = kwargs or {}
    if isinstance(dynamic_shapes, dict):
        dynamic_shapes = dynamic_shapes.values()  # type: ignore[assignment]
    dynamic_shapes = tuple(dynamic_shapes)
    combined_args = tuple(args) + tuple(kwargs.values())

    # run same check when we're processing shapes for export - is this too lazy?
    _check_dynamic_shapes(dict(enumerate(combined_args)), dynamic_shapes)  # type: ignore[arg-type]

    tree_shapes = _tree_map_with_path(
        _standardize_shapes, combined_args, dynamic_shapes, tree_name="inputs"
    )
    serialized_shapes = tree_map(_track_dim_from_dims, tree_shapes)
    return _postprocess_serialized_shapes(serialized_shapes, dims, to_dict=to_dict)


def _load_dynamic_shapes(
    spec: Union[DynamicShapesSpec, dict[str, Any]],
    from_dict: Optional[bool] = False,
) -> Union[dict[str, Any], tuple[Any], list[Any], None]:
    """
    Utility function for dynamic shapes serialization.
    Deserializes a DynamicShapesSpec or corresponding dictionary into a dynamic_shapes input to export().
    """
    import sympy

    from torch.fx.experimental.symbolic_shapes import _is_supported_equivalence

    if from_dict:
        if not isinstance(spec, dict):
            raise UserError(
                UserErrorType.INVALID_INPUT,
                f"With from_dict=True, expected `spec` to be a dict, got {type(spec)}",
            )
        if sorted(spec.keys()) != ["dims", "dynamic_shapes"]:
            raise UserError(
                UserErrorType.INVALID_INPUT,
                "With from_dict=True, expected `spec` to have keys `dims` and `dynamic_shapes`, "
                f"instead found {spec.keys()}",
            )
        dims = {}
        for k, v in spec["dims"].items():
            if not isinstance(k, str):
                raise UserError(
                    UserErrorType.INVALID_INPUT,
                    f"Expected `spec['dims']` keys to be strings for symbols, got key {type(k)}",
                )
            if sorted(v.keys()) != ["derived", "max", "min"]:
                raise UserError(
                    UserErrorType.INVALID_INPUT,
                    f"Expected `spec['dims']` values to have keys `derived`, `max`, and `min`, "
                    f"instead found {v.keys()}",
                )
            if not isinstance(v["min"], int):
                raise UserError(
                    UserErrorType.INVALID_INPUT,
                    f"Expected dims in `spec['dims']` to map `min` to an int, got {k}: {v['min']}",
                )
            if not isinstance(v["max"], int) or v["max"] is None:
                raise UserError(
                    UserErrorType.INVALID_INPUT,
                    f"Expected dims in `spec['dims']` to map `max` to an int or None, got {k}: {v['max']}",
                )
            if not isinstance(v["derived"], list) or any(
                not isinstance(d, str) for d in v["derived"]
            ):
                raise UserError(
                    UserErrorType.INVALID_INPUT,
                    "Expected dims in `spec['dims']` to map `derived` to a list of derived expressions, "
                    f"got {k}: {v['derived']}",
                )
            dims[k] = RootDim(**v)
        dynamic_shapes = spec["dynamic_shapes"]
    else:
        if not isinstance(spec, DynamicShapesSpec):
            raise UserError(
                UserErrorType.INVALID_INPUT,
                f"Expected `spec` to be a DynamicShapesSpec, got {type(spec)}",
            )
        dims = spec.dims
        dynamic_shapes = spec.dynamic_shapes

    if dynamic_shapes is None:
        return None

    dim_cache = {}
    for name, info in dims.items():
        symbol = sympy.sympify(name)
        if not isinstance(symbol, sympy.Symbol):
            raise UserError(
                UserErrorType.INVALID_INPUT,
                f"Expected `spec['dims']` keys to be symbols, got {name}",
            )
        dim_cache[name] = Dim(name, min=info.min, max=info.max)  # cache root dim
        for _expr in info.derived:
            expr = sympy.sympify(_expr)
            if len(expr.free_symbols) != 1 or symbol not in expr.free_symbols:
                raise UserError(
                    UserErrorType.INVALID_INPUT,
                    f"Expected derived expressions in to have {name} as the only free symbol, got {expr}",
                )
            if not _is_supported_equivalence(expr):
                raise UserError(
                    UserErrorType.INVALID_INPUT,
                    f"Expected derived expressions to be linear expressions, got {expr}",
                )
            modulus, remainder = sympy.polys.polytools.div(expr, symbol)
            ddim = dim_cache[name]
            if modulus != 1:
                ddim = int(modulus) * ddim
            if remainder != 0:
                ddim = ddim + int(remainder)
            dim_cache[_expr] = ddim  # cache derived dims

    def deserialize_shape(
        val: Union[None, int, str]
    ) -> Union[None, int, _Dim, _DimHint]:
        if val is None or isinstance(val, int):
            return val
        elif val == "_DimHint.AUTO":
            return _DimHint.AUTO
        elif val == "_DimHint.STATIC":
            return _DimHint.STATIC
        if not isinstance(val, str):
            raise UserError(
                UserErrorType.INVALID_INPUT,
                "Expected leaves in `spec['dynamic_shapes']` to be ints, None, Dim.AUTO/STATIC, symbols, "
                f" or derived expressions, got {val}",
            )
        if val not in dim_cache:
            raise UserError(
                UserErrorType.INVALID_INPUT,
                "Expected dims in `spec['dynamic_shapes']` to be tracked in `spec['dims']`, "
                f"got {val} which is not in {dims.keys()}",
            )
        return dim_cache[val]

    return tree_map(deserialize_shape, dynamic_shapes)
