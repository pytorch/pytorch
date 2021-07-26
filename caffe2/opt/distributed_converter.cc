#include "caffe2/opt/converter.h"

namespace caffe2 {
namespace {

using namespace nom;
using namespace nom::repr;

TRIVIAL_CONVERTER(Declare);
REGISTER_CONVERTER(Declare, DeclareConverter);

TRIVIAL_CONVERTER(Export);
REGISTER_CONVERTER(Export, ExportConverter);

} // namespace
} // namespace caffe2
