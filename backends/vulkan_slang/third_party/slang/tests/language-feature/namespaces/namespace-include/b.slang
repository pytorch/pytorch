implementing a;

import m;

namespace ns1
{
namespace ns2
{
    int k() { return f() + g(); } // should be able to find ns1::ns2::f in m.
}
}
