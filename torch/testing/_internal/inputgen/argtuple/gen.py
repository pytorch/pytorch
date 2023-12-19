from typing import Any, List, OrderedDict, Tuple

from inputgen.argtuple.engine import MetaArgTupleEngine
from inputgen.argument.engine import MetaArg
from inputgen.argument.gen import ArgumentGenerator
from inputgen.specs.model import Spec


class ArgumentTupleGenerator:
    def __init__(self, spec: Spec):
        self.spec = spec

    def gen_tuple(
        self, meta_tuple: Tuple[MetaArg], *, out: bool = False
    ) -> Tuple[List[Any], OrderedDict[str, Any]]:
        args = []
        kwargs = OrderedDict()
        for ix, arg in enumerate(self.spec.inspec):
            m = meta_tuple[ix]
            val = ArgumentGenerator(m).gen()
            if arg.kw:
                kwargs[arg.name] = val
            else:
                args.append(val)
        if out:
            for ix, arg in enumerate(self.spec.outspec):
                m = meta_tuple[ix + len(self.spec.inspec)]
                val = ArgumentGenerator(m).gen()
                kwargs[arg.name] = val
        return args, kwargs

    def gen(
        self, *, valid: bool = True, out: bool = False
    ) -> Tuple[List[Any], OrderedDict[str, Any]]:
        engine = MetaArgTupleEngine(self.spec, out=out)
        for meta_tuple in engine.gen(valid=valid):
            yield self.gen_tuple(meta_tuple, out=out)
