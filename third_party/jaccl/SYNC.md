# JACCL vendored source

`third_party/jaccl/jaccl/` is a verbatim copy of
[`mlx/distributed/jaccl/lib/jaccl/`](https://github.com/ml-explore/mlx/tree/main/mlx/distributed/jaccl/lib/jaccl)
from the [ml-explore/mlx](https://github.com/ml-explore/mlx) repository. We vendor
only the library source (not MLX's tensor runtime or integration layer) because
JACCL is designed to stand alone — the public API is just `#include <jaccl/jaccl.h>`.

## Current snapshot

- **Upstream commit:** `e8ebdebeeb655feaa85a51f6b24ece5b6d5518d1`
- **Upstream path:** `mlx/distributed/jaccl/lib/jaccl/`
- **License:** MIT (see `LICENSE`, copied from upstream `mlx/LICENSE`)

## Updating

To pull a newer JACCL:

```bash
tmp=$(mktemp -d)
git clone --depth 1 --filter=blob:none --sparse https://github.com/ml-explore/mlx.git "$tmp"
git -C "$tmp" sparse-checkout set mlx/distributed/jaccl/lib/jaccl LICENSE

rsync -a --delete "$tmp/mlx/distributed/jaccl/lib/jaccl/" third_party/jaccl/jaccl/
cp "$tmp/LICENSE" third_party/jaccl/LICENSE

# Record the commit you pulled from in this file.
git -C "$tmp" rev-parse HEAD
```

Do not edit files under `third_party/jaccl/jaccl/` directly — if a local fix is
needed, upstream it to ml-explore/mlx instead.
