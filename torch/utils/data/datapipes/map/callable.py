from torch.utils.data.datapipes.utils.common import check_lambda_fn
from typing import Callable, TypeVar
from torch.utils.data import MapDataPipe, functional_datapipe

T_co = TypeVar('T_co', covariant=True)


# Default function to return each item directly
# In order to keep datapipe picklable, eliminates the usage
# of python lambda function
def default_fn(data):
    return data


@functional_datapipe('map')
class MapperMapDataPipe(MapDataPipe[T_co]):
    r"""
    Apply the input function over each item from the source DataPipe (functional name: ``map``).
    The function can be any regular Python function or partial object. Lambda
    function is not recommended as it is not supported by pickle.

    Args:
        datapipe: Source MapDataPipe
        fn: Function being applied to each item
    """
    datapipe: MapDataPipe
    fn: Callable

    def __init__(
        self,
        datapipe: MapDataPipe,
        fn: Callable = default_fn,
    ) -> None:
        super().__init__()
        self.datapipe = datapipe
        check_lambda_fn(fn)
        self.fn = fn  # type: ignore[assignment]

    def __len__(self) -> int:
        return len(self.datapipe)

    def __getitem__(self, index) -> T_co:
        return self.fn(self.datapipe[index])
