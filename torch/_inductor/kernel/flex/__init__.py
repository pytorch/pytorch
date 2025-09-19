# mypy: allow-untyped-defs
# Import so here and then reimport above so that register_lowering gets triggered
from . import flex_attention, flex_decoding
