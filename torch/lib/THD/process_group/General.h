#pragma once

#include <string>
#include "../THD.h"
#include "../base/DataChannel.h"

THD_API void THDProcessGroupInit(
    THDChannelType channel_type,
    std::string init_method,
    int world_size,
    std::string group_name,
    int rank);
THD_API void THDProcessGroupDestroy();
THD_API void THDClearGroupCache(THDGroup group);
