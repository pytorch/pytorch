#include "DataChannelRequest.hpp"


THD_API void THDRequest_free(THDRequest* request) {
  delete request;
}
