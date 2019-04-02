// Client is a very thin wrapper over a Caffe2 interface, allowing us to do
// a very primitive caffe network call without the need of revealing all
// the header files inside Caffe2. Also, what we are going to deal with is
// always float inputs and float outputs, and the input and output shapes
// should be fixed. This is minimal and is only used by Yangqing to deal
// with quick demo cases.

#ifndef CAFFE2_CORE_CLIENT_H_
#define CAFFE2_CORE_CLIENT_H_

#include <string>
#include <vector>

namespace caffe2 {

// Forward declaration of a Caffe workspace.
class Blob;
class Workspace;

// Workspace is a class that holds all the blobs in this run and also runs
// the operators.
class Client {
 public:
  explicit Client(const std::string& client_def_name);
  ~Client();

  // TODO(Yangqing): Figure out how we can deal with different types of
  // inputs.
  bool Run(const std::vector<float>& input, std::vector<float>* output);

 private:
  // TODO(Yangqing): Are we really going to share workspaces? If not, let's
  // remove this unnecessity.
  Workspace* workspace_;
  Blob* input_blob_;
  Blob* output_blob_;
};

}  // namespace caffe2

#endif  // CAFFE2_CORE_CLIENT_H_
