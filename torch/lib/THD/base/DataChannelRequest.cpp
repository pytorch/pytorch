#include "DataChannelRequest.hpp"

THD_API void THDRequest_free(void* request) {
  delete (THDRequest*)request;
}
