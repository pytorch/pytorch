#ifndef PYTORCH_AUTOGRAD_ERRORS_H
#define PYTORCH_AUTOGRAD_ERRORS_H

#define PT_ERR_BACKWARD_TWICE "Trying to backward through the " \
      "graph second time, but the buffers have already been freed. Please " \
      "specify retain_variables=True when calling backward for the first time."


#endif
