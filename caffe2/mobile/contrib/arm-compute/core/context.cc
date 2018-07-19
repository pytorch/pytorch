#include "context.h"

#include "caffe2/core/allocator.h"
#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/tensor.h"

namespace caffe2 {

CAFFE_KNOWN_TYPE(GLTensor<GLfloat>);
CAFFE_KNOWN_TYPE(GLTensor<GLhalf>);
CAFFE_KNOWN_TYPE(GLTensor<half>);
CAFFE_KNOWN_TYPE(Tensor<GLContext>);

bool GLContext::initialized = false;

GLContext::GLContext() {
  CAFFE_ENFORCE(arm_compute::opengles31_is_available());
  if(!initialized) {
    arm_compute::GCScheduler::get().default_init();
    initialized = true;
  }
}

void EventCreateOPENGL(const DeviceOption & /* unused */,
                       Event * /* unused */) {}
void EventRecordOPENGL(Event * /* unused */, const void * /* unused */,
                       const char * /* unused */) {}
void EventWaitOPENGLOPENGL(const Event * /* unused */, void * /* unused */) {}
void EventFinishOPENGL(const Event * /* unused */) {}
void EventResetOPENGL(Event * /* unused */) {}

REGISTER_EVENT_CREATE_FUNCTION(OPENGL, EventCreateOPENGL);
REGISTER_EVENT_RECORD_FUNCTION(OPENGL, EventRecordOPENGL);
REGISTER_EVENT_WAIT_FUNCTION(OPENGL, OPENGL, EventWaitOPENGLOPENGL);
REGISTER_EVENT_FINISH_FUNCTION(OPENGL, EventFinishOPENGL);
REGISTER_EVENT_RESET_FUNCTION(OPENGL, EventResetOPENGL);

} // namespace caffe2
