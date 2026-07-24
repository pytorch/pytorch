# Torch-free pointwise row data: the single source of truth for the op
# table, readable at EVERY build stage. table.py (JIT registration,
# torch importable) rebuilds typed PointwiseDef rows from these tuples;
# aot.py (torchgen stage 1, stdlib only) file-path-loads this module
# directly. Fields mirror PointwiseDef with promotion as a NAME and the
# two callable escape hatches reduced to flags/names:
#   (aten, nin, fn, promotion, scalars, nout, out_dtypes_tag, dtypes_tag)
# out_dtypes_tag: None | "frexp" (mantissa + int32 exponent).
# dtypes_tag: None | "no_fp64" (frexp's log2 trick is inexact in fp64).

PW_ROWS = (
    # --- binary / unary arithmetic (DEFAULT promotion) ---
    ("neg", 1, "_neg", "DEFAULT", (), 1, None, None),
    ("add.Tensor", 2, "_add", "DEFAULT", ("alpha",), 1, None, None),
    ("sub.Tensor", 2, "_sub", "DEFAULT", ("alpha",), 1, None, None),
    ("mul.Tensor", 2, "_mul", "DEFAULT", (), 1, None, None),
    ("div.Tensor", 2, "_div", "DEFAULT", (), 1, None, None),
    ("maximum", 2, "_maximum", "DEFAULT", (), 1, None, None),
    ("minimum", 2, "_minimum", "DEFAULT", (), 1, None, None),
    ("atan2", 2, "_atan2", "DEFAULT", (), 1, None, None),
    # --- rounding / sign / activation (DEFAULT) ---
    ("floor", 1, "_floor", "DEFAULT", (), 1, None, None),
    ("ceil", 1, "_ceil", "DEFAULT", (), 1, None, None),
    ("trunc", 1, "_trunc", "DEFAULT", (), 1, None, None),
    ("sign", 1, "_sign", "DEFAULT", (), 1, None, None),
    ("relu", 1, "_relu", "DEFAULT", (), 1, None, None),
    # --- unary transcendental (INT_TO_FLOAT) ---
    ("exp", 1, "_exp", "INT_TO_FLOAT", (), 1, None, None),
    ("exp2", 1, "_exp2", "INT_TO_FLOAT", (), 1, None, None),
    ("expm1", 1, "_expm1", "INT_TO_FLOAT", (), 1, None, None),
    ("log", 1, "_log", "INT_TO_FLOAT", (), 1, None, None),
    ("log2", 1, "_log2", "INT_TO_FLOAT", (), 1, None, None),
    ("log10", 1, "_log10", "INT_TO_FLOAT", (), 1, None, None),
    ("log1p", 1, "_log1p", "INT_TO_FLOAT", (), 1, None, None),
    ("sqrt", 1, "_sqrt", "INT_TO_FLOAT", (), 1, None, None),
    ("rsqrt", 1, "_rsqrt", "INT_TO_FLOAT", (), 1, None, None),
    ("reciprocal", 1, "_reciprocal", "INT_TO_FLOAT", (), 1, None, None),
    ("sin", 1, "_sin", "INT_TO_FLOAT", (), 1, None, None),
    ("cos", 1, "_cos", "INT_TO_FLOAT", (), 1, None, None),
    ("tan", 1, "_tan", "INT_TO_FLOAT", (), 1, None, None),
    ("asin", 1, "_asin", "INT_TO_FLOAT", (), 1, None, None),
    ("acos", 1, "_acos", "INT_TO_FLOAT", (), 1, None, None),
    ("atan", 1, "_atan", "INT_TO_FLOAT", (), 1, None, None),
    ("tanh", 1, "_tanh", "INT_TO_FLOAT", (), 1, None, None),
    ("erf", 1, "_erf", "INT_TO_FLOAT", (), 1, None, None),
    ("sigmoid", 1, "_sigmoid", "INT_TO_FLOAT", (), 1, None, None),
    # --- comparisons (ALWAYS_BOOL) ---
    ("gt.Tensor", 2, "_gt", "ALWAYS_BOOL", (), 1, None, None),
    ("lt.Tensor", 2, "_lt", "ALWAYS_BOOL", (), 1, None, None),
    ("ge.Tensor", 2, "_ge", "ALWAYS_BOOL", (), 1, None, None),
    ("le.Tensor", 2, "_le", "ALWAYS_BOOL", (), 1, None, None),
    ("eq.Tensor", 2, "_eq", "ALWAYS_BOOL", (), 1, None, None),
    ("ne.Tensor", 2, "_ne", "ALWAYS_BOOL", (), 1, None, None),
    # --- ternary / multi-output ---
    ("addcmul", 3, "_addcmul", "DEFAULT", ("value",), 1, None, None),
    ("frexp.Tensor", 1, "_frexp", "DEFAULT", (), 2, "frexp", "no_fp64"),
)
