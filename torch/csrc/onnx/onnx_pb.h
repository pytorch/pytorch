#pragma once

// Protobuf includes Abseil logging, whose macros collide with c10 logging.
#pragma push_macro("LOG")
#pragma push_macro("VLOG")
#pragma push_macro("LOG_IF")
#pragma push_macro("DLOG")
#undef LOG
#undef VLOG
#undef LOG_IF
#undef DLOG
#include <onnx/onnx_pb.h>
#undef LOG
#undef VLOG
#undef LOG_IF
#undef DLOG
#pragma pop_macro("DLOG")
#pragma pop_macro("LOG_IF")
#pragma pop_macro("VLOG")
#pragma pop_macro("LOG")
