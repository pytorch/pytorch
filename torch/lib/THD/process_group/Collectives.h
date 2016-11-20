#pragma once

#include "../THD.h"

THD_API int THDGetRank();
THD_API int THDGetNumProcesses();
THD_API void THDAllReduce(THDTensorDescriptor desc);
THD_API void THDReduce(THDTensorDescriptor desc, int dst_rank);
THD_API void THDBroadcast(THDTensorDescriptor desc, int src_rank);
THD_API void THDSend(THDTensorDescriptor desc, int dst_rank);
THD_API void THDReceive(THDTensorDescriptor desc, int src_rank);
