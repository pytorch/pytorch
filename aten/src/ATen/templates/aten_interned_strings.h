#pragma once

// ${generated_comment}

#if defined(TORCH_ASSERT_NO_OPERATORS) || defined(TORCH_ASSERT_ONLY_METHOD_OPERATORS)
#error This change adds a dependency on native_functions.yaml,          \
  meaning the file will need to be re-compiled every time an operator   \
  is changed or added. Consider if including <ATen/core/symbol.h> for   \
  the c10::Symbol class would be sufficient, or if your change would be \
  better placed in another file.
#endif

// ATen symbols correspond exactly to operators defined in ATen. Every
// symbol here corresponds exactly to an ATen operation defined in
// native_functions.yaml; attributes are in one-to-one correspondence
// with their ATen name.

#define FORALL_ATEN_BASE_SYMBOLS(_) \
${aten_symbols}

#define FORALL_ATTR_BASE_SYMBOLS(_) \
${attr_symbols}
