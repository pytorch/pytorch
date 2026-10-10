from __future__ import annotations

import dataclasses


class Foo:
    pass


@dataclasses.dataclass
class Bar:
    foo: Foo
