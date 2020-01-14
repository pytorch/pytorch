#include <c10/util/TypeIndex.h>

namespace c10 {
namespace util {
namespace detail {

CompilerClashChecker& CompilerClashChecker::singleton() {
    static CompilerClashChecker singleton;
    return singleton;
}

}
}
}
