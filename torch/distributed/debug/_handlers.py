import tempfile
import time

from torch._C._distributed_c10d import _register_handler, _Request, _Response
from torch.profiler import _ExperimentalConfig, profile


def _torch_profile(req: _Request, resp: _Response) -> None:
    experimental_config = _ExperimentalConfig(
        profile_all_threads=True,
    )
    duration = float(req.get_param("duration"))
    with profile(record_shapes=True, experimental_config=experimental_config) as prof:
        time.sleep(duration)

    with tempfile.NamedTemporaryFile(prefix="torch_debug", suffix=".json") as f:
        prof.export_chrome_trace(f.name)
        resp.set_content(open(f.name, "rb").read(), "application/json")
        resp.set_status(200)


_register_handler("torch_profile", _torch_profile)
