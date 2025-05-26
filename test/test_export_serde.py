import io
import torch
from torch._export.serde.serialize import deserialize_torch_artifact

def test_deserialize_torch_artifact_dict():
    data = {"key": torch.tensor([1, 2, 3])}
    buf = io.BytesIO()
    torch.save(data, buf)
    serialized = buf.getvalue()

    result = deserialize_torch_artifact(serialized)
    assert isinstance(result, dict)
    assert torch.equal(result["key"], torch.tensor([1, 2, 3]))
