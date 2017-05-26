#pragma once

#include "../../THD.h"
#include <string>

THD_API bool THDMasterWorkerInit(THDChannelType channel_type, std::string init_method = "env://",
                                 int world_size = -1, std::string group_name = "");
