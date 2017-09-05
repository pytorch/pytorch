#pragma once

#include "process_group/General.hpp"

#include <THPP/Traits.hpp>

template<typename T>
T receiveValueFromWorker(int worker_id) {
  thpp::Type type = thpp::type_traits<T>::type;
  if (thpp::isInteger(type)) {
    thd::IntScalar wrapped_value;
    thd::dataChannel->receive(wrapped_value, worker_id);
    return static_cast<T>(wrapped_value.value());
  } else if (thpp::isFloat(type)) {
    thd::FloatScalar wrapped_value;
    thd::dataChannel->receive(wrapped_value, worker_id);
    return static_cast<T>(wrapped_value.value());
  } else {
    throw std::invalid_argument("expected scalar type");
  }
}
