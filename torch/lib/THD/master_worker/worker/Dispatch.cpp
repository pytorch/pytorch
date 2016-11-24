#include <cstdint>
#include <unordered_map>
#include <memory>
#include <string>
#include <unordered_map>

#include "../common/Functions.hpp"
#include "../common/RPC.hpp"

namespace thd {
namespace worker {


std::string execute(std::unique_ptr<rpc::RPCMessage> raw_message) {
  try {
    // TODO: call function from the map
    return std::string();
  } catch(std::exception& e) {
    return std::string(e.what());
  }
}


namespace detail {

static void add(const std::string& raw_message) {
//THTensor& result = parse_tensor(raw_message);
  //THTensor& source = parse_tensor(raw_message);
  //double x = parse_scalar(raw_message);
  //assert_end(raw_message);
  //result.add(source, x);
}

using dispatch_fn = void (*)(const std::string&);
using Functions = thd::Functions;


static const std::unordered_map<std::uint16_t, dispatch_fn> functions {
    {Functions::add, add}
};

} // namespace detail


} // namespace thd
} // namespace worker
