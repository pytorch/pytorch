#include <ATen/native/RNN.h>
#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/InitialTensorOptions.h>
#include <ATen/MatrixRef.h>
#include <ATen/NativeFunctions.h>
#include <ATen/TensorUtils.h>

#include <ATen/cuda/CUDAConfig.h>
#include <ATen/cuda/CUDAEvent.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/util/Exception.h>

#if !AT_ROCM_ENABLED()

namespace at { namespace native {

    Tensor miopen_rnn_flatten_weight(
        TensorList weight_arr, int64_t weight_stride0, int64_t input_size,
        int64_t fn_mode, int64_t fn_hidden_size, int64_t fn_num_layers,
        bool batch_first, bool fn_bidirectional
        ) {
      AT_ERROR("miopen_flatten_weight: ATen not compiled with MIOpen support.");
    }

    std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor> miopen_rnn(
        const Tensor& input_r, TensorList weight, int64_t weight_stride0,
        const Tensor& weight_buf_r, const Tensor& hx, const Tensor& cx,
        int64_t fn_mode, int64_t fn_hidden_size, int64_t fn_num_layers,
        bool batch_first, double fn_dropout, bool fn_train, bool fn_bidirectional,
        IntArrayRef fn_batch_sizes, const Tensor& fn_dropout_state
        ) {
      AT_ERROR("miopen_rnn : ATen not compiled with MIOpen support.");
    }

    std::tuple<Tensor, Tensor, Tensor, std::vector<Tensor>> miopen_rnn_backward(
        const Tensor& input, TensorList weight, int64_t weight_stride0, const Tensor& weight_buf, const Tensor& hx, const Tensor& cx,
        const Tensor& output, const Tensor& grad_output_r, const Tensor& grad_hy_r,
        const Tensor& grad_cy_r, int64_t mode, int64_t hidden_size, int64_t num_layers, bool batch_first,
        double dropout, bool train, bool bidirectional, IntArrayRef batch_sizes, const Tensor& dropout_state, 
        const Tensor& reserve, std::array<bool, 4> output_mask
        ) {
      AT_ERROR("miopen_rnn_backward: ATen not compiled with MIOpen support.");
    }

}} //namespace at::native

#else // AT_CUDNN_ENABLED()

	//RNNDescriptor.
	struct RNNDescriptorParams {
		int64_t hidden_size;
		int64_t num_layers;
		miopenRNNDirectionMode_t direction;
		miopenRNNMode_t rnn_mode;
		miopenDataType_t datatype;
		miopenRNNAlgo_t algo = miopenRNNdefault;
		miopenRNNInputMode_t input_mode = miopenRNNlinear;

		void set_bidirectional(bool fn_bidirectional) {
			direction = fn_bidirectional ? miopenRNNbidirection : miopenRNNunidirection;
		}

		void set_algo(miopenRNNAlgo_t algo) {
			this->algo = algo;
		}

		/*fn_mode is set in torch.backends.cudnn (get_cudnn_mode() method) 
	  	  Need to modify the interface to the frontend to make this function useful.
		*/
    	void set_mode(int64_t fn_mode) {
      		switch (fn_mode) {
        		case 0:
          			rnn_mode = miopenRNNRELU;
          			break;
        		case 1:
          			rnn_mode = miopenRNNTANH;
          			break;
        		case 2:
          			rnn_mode = miopenLSTM;
          			break;
        		case 3:
          			rnn_mode = miopenGRU;
          			break;
        		default:
        		{
          			std::ostringstream oss;
          			oss << "unrecognized miopen RNN mode " << fn_mode;
          			AT_ERROR(oss.str());
        		}
      	}
    }

	void set(int64_t mode, int64_t hidden_size, int64_t num_layers, bool bidirectional, miopenDataType_t datatype) {
			this->set_mode(mode);
			this->hidden_size = hidden_size;
			this->num_layers = num_layers;
			this->direction = this->set_bidirectional(bidirectional);
			this->datatype = datatype;
		}
	}

  	Tensor miopen_rnn_flatten_weight(
        TensorList weight_arr, int64_t weight_stride0, int64_t input_size,
        miopenRNNMode_t fn_mode, int64_t fn_hidden_size, int64_t fn_num_layers,
        bool batch_first, bool fn_bidirectional
        ) {
      AT_ERROR("miopen_flatten_weight: not implemented yet.");
  	  
  	  AT_CHECK(weight_arr.size() > 0, "miopen_rnn_flatten_weight : cannot flatten empty weight list.");

  	  auto any_param = weight_arr[0];
  	  auto datatype = getMiopenDataType(any_param);

  	  /*TODO: 

  	  	[Done]1. Need to create RNNDescriptorParam and set RNN Descriptor.
  	  	[Done]2. Create RNNDescriptors in Descriptors.h
		3. set the RNNDescriptor in RNNDescriptors.
		4. Flatten the weights.		
  	  */

  	  RNNDescriptorParam rnn;
  	  rnn.set(fn_mode, hidden_size, num_layers, bidirectional, datatype);

    }

    std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor> miopen_rnn(
        const Tensor& input_r, TensorList weight, int64_t weight_stride0,
        const Tensor& weight_buf_r, const Tensor& hx, const Tensor& cx,
        int64_t fn_mode, int64_t fn_hidden_size, int64_t fn_num_layers,
        bool batch_first, double fn_dropout, bool fn_train, bool fn_bidirectional,
        IntArrayRef fn_batch_sizes, const Tensor& fn_dropout_state
        ) {
      AT_ERROR("miopen_rnn : not implemented yet.");
    }

    std::tuple<Tensor, Tensor, Tensor, std::vector<Tensor>> miopen_rnn_backward(
        const Tensor& input, TensorList weight, int64_t weight_stride0, const Tensor& weight_buf, const Tensor& hx, const Tensor& cx,
        const Tensor& output, const Tensor& grad_output_r, const Tensor& grad_hy_r,
        const Tensor& grad_cy_r, int64_t mode, int64_t hidden_size, int64_t num_layers, bool batch_first,
        double dropout, bool train, bool bidirectional, IntArrayRef batch_sizes, const Tensor& dropout_state, 
        const Tensor& reserve, std::array<bool, 4> output_mask
        ) {
      AT_ERROR("miopen_rnn_backward: not implemented yet.");
    }


#endif 
