// ${generated_comment}

#ifndef PYTORCH_CODEGENFUNCTIONS_H
#define PYTORCH_CODEGENFUNCTIONS_H
#include <ATen/ATen.h>
namespace at {
namespace unboxing {
namespace {

template<typename T, size_t N>
std::array<T, N> as_array(const c10::List<c10::IValue>& list) {
    std::array<T, N> res;
    AT_ASSERT(list.size() == N);
    std::vector<T> vec;
    for (c10::IValue elem : list) {
        vec.push_back(elem.to<T>());
    }
    std::copy(vec.begin(), vec.end(), res.begin());
    return res;
}
}  // namespace <anonymous>
using Stack = std::vector<c10::IValue>;
// Generated function declaration
${declarations}

} // namespace unboxing
} // namespace at


#endif //PYTORCH_CODEGENFUNCTIONS_H
