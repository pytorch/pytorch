
#include "GLPredictor.h"
#include "GLContext.h"
#include "rewrite_net.h"
#include <vector>

namespace caffe2 {

template <class T>
void shareInputGLImage(Workspace* ws, const std::string& name, GLImageVector<T>* input) {
  auto* blob = ws->GetBlob(name);
  CAFFE_ENFORCE(blob, "Blob: ", name, " does not exist");
  blob->ShareExternal<GLImageVector<T>>(input);
}

template <class T>
const GLImageVector<T>* extractOutputGLImage(Workspace* ws, const std::string& name) {
  auto* blob = ws->GetBlob(name);
  CAFFE_ENFORCE(blob, "Blob: ", name, " does not exist");
  return &blob->template Get<GLImageVector<T>>();
}

const NetDef create_gl_run_net(const NetDef& init_net,
                               const NetDef& run_net,
                               bool use_texture_input) {
  NetDef gl_run_net;
  if (!tryConvertToOpenGL(init_net, run_net, &gl_run_net, use_texture_input)) {
    CAFFE_THROW("Failed to convert model to OpenGL");
  }
  return gl_run_net;
}

GLPredictor::GLPredictor(const NetDef& init_net,
                         const NetDef& run_net,
                         bool use_texture_input,
                         Workspace* parent)
    : Predictor(init_net, create_gl_run_net(init_net, run_net, use_texture_input), parent) {}

GLPredictor::~GLPredictor() {}

template <class T>
bool GLPredictor::run(std::vector<GLImageVector<T>*>& inputs,
                      std::vector<const GLImageVector<T>*>* outputs) {
  const NetDef& run_net_ = Predictor::def();
  CAFFE_ENFORCE(inputs.size() <= run_net_.external_input_size());
  for (auto i = 0; i < inputs.size(); ++i) {
    shareInputGLImage<T>(Predictor::ws(), run_net_.external_input(i), inputs[i]);
  }

  if (!Predictor::ws()->RunNet(run_net_.name())) {
    return false;
  }

  for (auto i = 0; i < run_net_.external_output_size(); ++i) {
    outputs->push_back(extractOutputGLImage<T>(Predictor::ws(), run_net_.external_output(i)));
  }

  return true;
}

template bool GLPredictor::run(std::vector<GLImageVector<uint8_t>*>& inputs,
                               std::vector<const GLImageVector<uint8_t>*>* outputs);
} // namespace caffe2
