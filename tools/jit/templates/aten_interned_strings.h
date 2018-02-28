#pragma once

// ${generated_comment}

// ATen symbols correspond exactly to operators defined in ATen.  Every
// symbol here corresponds exactly to an ATen operation which is defined
// in Declarations.yaml; attributes are in one-to-one correspondence with
// their ATen name.

#define FORALL_ATEN_SYMBOLS(_) \
${aten_symbols}
_(__ATEN_END)
