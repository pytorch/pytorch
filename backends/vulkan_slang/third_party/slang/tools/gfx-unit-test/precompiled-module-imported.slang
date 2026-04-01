module "precompiled-module-imported";

__include "precompiled-module-included.slang";

namespace ns
{
    public int helperFunc()
    {
        return 1;
    }
}