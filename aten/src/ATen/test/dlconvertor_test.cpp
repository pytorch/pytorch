#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "ATen/ATen.h"
#include "ATen/DLConvertor.h"

#include <iostream>
#include <string.h>
#include <sstream>
#include "test_seed.h"

using namespace at;

TEST_CASE( "dlconvertor", "[cpu]" ) {

  manual_seed(123, at::Backend::CPU);

  INFO( "convert ATen to DLTensor" );

  Tensor a = rand(CPU(at::kFloat), {3,4});
  DLManagedTensor* dlMTensor = toDLPack(a);

  INFO( "convert DLTensor to ATen" );
  Tensor b = fromDLPack(dlMTensor);

  REQUIRE(a.equal(b));
}

