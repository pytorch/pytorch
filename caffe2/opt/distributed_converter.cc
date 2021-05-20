#include "caffe2/opt/converter.h"

namespace caffe2 {
namespace {

using namespace nom;
using namespace nom::repr;

TRIVIAL_CONVERTER(Declare);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CONVERTER(Declare, DeclareConverter);

TRIVIAL_CONVERTER(Export);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CONVERTER(Export, ExportConverter);

} // namespace
} // namespace caffe2
