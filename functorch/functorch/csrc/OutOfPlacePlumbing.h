#pragma once


namespace at { namespace functorch {

template <typename batch_rule_t, typename Result, typename... Args>
Result lowerToNextLayer(batch_rule_t batch_rule, Args... args);

}}
