#include <ATen/core/op_registration/op_registration.h>

#ifndef C10_MOBILE

namespace c10 {

RegisterOperators::RegisterOperators() = default;
RegisterOperators::~RegisterOperators() = default;
RegisterOperators::RegisterOperators(RegisterOperators&&) noexcept = default;
RegisterOperators& RegisterOperators::operator=(RegisterOperators&&) = default;

}

#endif
