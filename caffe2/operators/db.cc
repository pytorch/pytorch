#include "caffe2/operators/db.h"

namespace caffe2 {
namespace db {

DEFINE_REGISTRY(Caffe2DBRegistry, DB, const string&, Mode);

}  // namespacd db
}  // namespace caffe2
