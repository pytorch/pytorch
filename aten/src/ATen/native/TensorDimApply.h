namespace at {
  namespace native {
    #define TENSOR_DIM_APPLY3(type1, tensor1, type2, tensor2, type3, tensor3, dim, code) \
    { \
      dim = maybe_wrap_dim(dim, tensor1.dim()); \
      int ndims = tensor1.dim();\
      if(ndims == 0) { \
        tensor2.fill_(tensor1); \
        tensor3.fill_(0); \
      } \
      else if(tensor1.numel() != 0) { \
        int tensor_dim_apply_has_finished = 0; \
        int tensor_dim_apply_i; \
        int TENSOR_DIM_APPLY_i;\
        int64_t tensor_dim_apply_counter[ndims];\
        for(tensor_dim_apply_i = 0; tensor_dim_apply_i < ndims; tensor_dim_apply_i++) \
          tensor_dim_apply_counter[tensor_dim_apply_i] = 0; \
      \
        type1* tensor1##_data = tensor1.data_ptr<type1>();\
        type2* tensor2##_data = tensor2.data_ptr<type2>();\
        type3* tensor3##_data = tensor3.data_ptr<type3>();\
        int64_t tensor1##_stride = tensor1.stride(dim);\
        int64_t tensor2##_stride = tensor2.stride(dim);\
        int64_t tensor3##_stride = tensor3.stride(dim);\
        while(!tensor_dim_apply_has_finished) \
        { \
          code \
        \
          if(ndims == 1) \
             break; \
        \
          for(tensor_dim_apply_i = 0; tensor_dim_apply_i < ndims; tensor_dim_apply_i++) \
          { \
            if(tensor_dim_apply_i == dim) \
            { \
              if(tensor_dim_apply_i == (ndims - 1)) \
              { \
                tensor_dim_apply_has_finished = 1; \
                break; \
              } \
              continue; \
            } \
            tensor_dim_apply_counter[tensor_dim_apply_i]++; \
            tensor1##_data += tensor1.stride(tensor_dim_apply_i); \
            tensor2##_data += tensor2.stride(tensor_dim_apply_i); \
            tensor3##_data += tensor3.stride(tensor_dim_apply_i); \
            \
            if(tensor_dim_apply_counter[tensor_dim_apply_i] == tensor1.size(tensor_dim_apply_i)) \
            { \
              if(tensor_dim_apply_i == ndims-1) \
              { \
                tensor_dim_apply_has_finished = 1; \
                break; \
              } \
              else \
              { \
                tensor1##_data -= tensor_dim_apply_counter[tensor_dim_apply_i]*tensor1.stride(tensor_dim_apply_i); \
                tensor2##_data -= tensor_dim_apply_counter[tensor_dim_apply_i]*tensor1.stride(tensor_dim_apply_i); \
                tensor3##_data -= tensor_dim_apply_counter[tensor_dim_apply_i]*tensor3.stride(tensor_dim_apply_i); \
                tensor_dim_apply_counter[tensor_dim_apply_i] = 0; \
              } \
            } \
            else \
              break; \
          } \
        } \
      }\
    }
  }
}
