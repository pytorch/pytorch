import torch.testing._internal.inputgen.specs.function as fn
from torch.testing._internal.inputgen.specs.model import ConstraintProducer as cp


DimDefault = [
    cp.Value.Ge(lambda deps: -deps[0].dim() if deps[0].dim() > 0 else None),
    cp.Value.Ge(lambda deps: -1 if deps[0].dim() == 0 else None),
    cp.Value.Le(lambda deps: deps[0].dim() - 1 if deps[0].dim() > 0 else None),
    cp.Value.Le(lambda deps: 0 if deps[0].dim() == 0 else None),
]

DimListDefault = [
    cp.Length.Le(lambda deps: deps[0].dim() if deps[0].dim() > 0 else None),
    cp.Length.Le(lambda deps: 1 if deps[0].dim() == 0 else None),
    cp.Value.Gen(
        lambda deps, length: (
            fn.valid_dim_list(deps[0], length),
            fn.invalid_dim_list(deps[0], length),
        )
    ),
]

IndexDefault = [
    cp.Value.Ge(lambda deps: -fn.safe_size(deps[0], deps[1])),
    cp.Value.Le(lambda deps: fn.safe_size(deps[0], deps[1]) - 1),
]

ShapeDefault = [cp.Value.Ge(lambda deps, length, ix: 0)]

MemoryFormatDefault = [cp.Value.In(lambda deps: [None])]
