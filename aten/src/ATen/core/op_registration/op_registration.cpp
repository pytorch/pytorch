#include <ATen/core/op_registration/op_registration.h>

namespace c10 {

RegisterOperators::RegisterOperators() = default;
RegisterOperators::~RegisterOperators() = default;
RegisterOperators::RegisterOperators(RegisterOperators&&) noexcept = default;
RegisterOperators& RegisterOperators::operator=(RegisterOperators&&) = default;

}
