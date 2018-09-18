#define CATCH_CONFIG_MAIN
#include "catch_utils.hpp"

#include "ATen/ATen.h"
#include "ATen/DLConvertor.h"

#include <iostream>
#include <string.h>
#include <sstream>
#include "test_seed.h"

using namespace at;

CATCH_TEST_CASE( "dlconvertor", "[cpu]" ) {

  manual_seed(123, at::kCPU);

  CATCH_INFO( "convert ATen to DLTensor" );

  Tensor a = rand({3,4});
  DLManagedTensor* dlMTensor = toDLPack(a);

  CATCH_INFO( "convert DLTensor to ATen" );
  Tensor b = fromDLPack(dlMTensor);

  CATCH_REQUIRE(a.equal(b));
}
