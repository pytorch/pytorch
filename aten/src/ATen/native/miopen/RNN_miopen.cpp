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
    miopenBiasMode_t bias_mode = miopenRNNNoBias;

    int64_t num_directions() const {
    	return (direction == miopenRNNbidirection) ? 2 : 1;
    }

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

    void set(int64_t mode, int64_t hidden_size, int64_t num_layers, bool bidirectional, miopenDataType_t datatype, miopenBiasMode_t bias_mode) {
        this->set_mode(mode);
        this->hidden_size = hidden_size;
        this->num_layers = num_layers;
        this->direction = this->set_bidirectional(bidirectional);
        this->datatype = datatype;
        this->bias_mode = bias_mode;
    }

    RNNDescriptor descriptor() const {
        RNNDescriptor rnn_desc;
        rnn_desc.set(hidden_size, num_layers, input_mode, direction, rnn_mode, bias_mode, algo, datatype);
        return rnn_desc;
    }
};

//TensorDescriptor list.
std::vector<TensorDescriptor> rnn_descriptor_sequence(const Tensor& tensor, IntArrayRef batch_sizes) {
	std::vector<TensorDescriptor> descriptors(batch_sizes.size());
	size_t i =0;

	auto batch_tensor_size = tensor.sizes().vec();
	for (auto batch_size : batch_sizes) {
		batch_tensor_size[0] = batch_size;

		descriptors[i].set(getMiopenDataType(tensor), batch_tensor_size, tensor.strides(), 3);
		i++;
	}

	return descriptors;
}

std::vector<TensorDescriptor> rnn_descriptor(const Tensor& tensor, int64_t N) {
	std::vector<TensorDescriptor> descriptors(N);
	for (int64_t i = 0; i < N ; i++) {
		descriptors[i].set(tensor, 5);
	}

	return descriptors;
}

struct TensorDescriptorListParams {
	IntArrayRef batch_sizes;
	int64_t batch_sizes;
	int64_t seq_length;
	int64_t mini_batch;

	int64_t input_size;
	int64_t batch_sizes_sum;

	bool is_input_packed() const {
		return batch_sizes.size() != 0;
	}

	void set(IntArrayRef input_sizes, IntArrayRef batch_sizes_, bool batch_first) {
		batch_sizes = batch_sizes_;
		if (is_input_packed()) {
			seq_length = batch_sizes.size();
			mini_batch = batch_sizes[0];
			batch_sizes_sum = input_sizes[0];
			input_size = input_sizes[1];
		} else {
			if (batch_first) {
				seq_length = input_sizes[1];
				mini_batch = input_sizes[0];
			} else {
				seq_length = input_sizes[0];
				mini_batch = input_sizes[1];
			}
			input_size = input_sizes[2];
			batch_sizes_sum = -1;
		}
	}

	std::vector<TensorDescriptor> descriptors(Tensor x) const {
		auto is_input_packed = batch_sizes.size() != 0;
		if (is_input_packed) {
			return rnn_descriptor_sequence(x, batch_sizes);
		} else {
			return rnn_descriptor(x[0], seq_length);
		}
	}
};

struct RNNParams {
	RNNDescriptorParams rnn;
	TensorDescriptorListParams tensors;
};

struct RNNDescriptors {
	RNNDescriptor rnn_desc;
	std::vector<TensorDescriptor> x_descs;
	std::vector<TensorDescriptor> y_descs;
	TensorDescriptor hx_desc;
	TensorDescriptor hy_desc;
	TensorDescriptor cx_desc;
	TensorDescriptor cy_desc;

	RNNDescriptors(const RNNParams& fn, miopenHandle_t handle, Tensor x, Tensor y, Tensor hx, Tensor cx) {
		rnn_desc = fn.rnn.descriptor();
		x_descs = fn.tensors.descriptors(x);
		y_descs = fn.tensors.descriptors(y);
		hx_desc.set(hx, 5);
		hy_desc.set(hy, 5);
		if (cx.defined()) {
			cx_desc.set(cx, 5);
			cy_desc.set(cy, 5);
		}
	}

	std::vector<miopenTensorDescriptor_t> get_descs(const std::vector<TensorDescriptor>& descs) {
		std::vector<miopenTensorDescriptor_t> r;
		r.reserve(descs.size());
		for (auto& desc : descs) {
			r.emplace_back(desc.desc());
		}
		return r;
	}

	std::vector<miopenTensorDescriptor_t> get_x_descs() {
		return get_descs(x_descs);
	}

	std::vector<miopenTensorDescriptor_t> get_y_descs() {
		return get_descs(y_descs);
	}
};

void _viewOrCopyParams(MatrixRef<Tensor> params_from, MatrixRef<Tensor> params_to, bool copy) {
    AT_ASSERTM(params_from.size(0) == params_to.size(0), "number of layers mismatch");
    for (size_t i = 0; i < params_from.size(0); i++) {
        auto layer_params_from = params_from[i];
        auto layer_params_to = params_to[i];
        // NOTE: these lists have all weights before all biases, so if the layer
        // doesn't use biases, iteration will terminate once layer_params_from ends
        // and ignore them.
        for (auto a = layer_params_from.begin(), b = layer_params_to.begin();
                a != layer_params_from.end() && b != layer_params_to.end();
                ++a, ++b) {
            auto param_from = *a, param_to = *b;
            AT_ASSERTM(param_from.type() == param_to.type(), "parameter types mismatch");
            if (copy) {
                param_to.copy_(param_from.view_as(param_to));
            } else {
                param_from.resize_as_(param_to);
            }
        }
    }
}

void _copyParams(MatrixRef<Tensor> params_from, MatrixRef<Tensor> params_to) {
    _viewOrCopyParams(params_from, params_to, true);
}

void _viewParams(MatrixRef<Tensor> params_from, MatrixRef<Tensor> params_to) {
    _viewOrCopyParams(params_from, params_to, false);
}

int64_t get_num_weights(miopenHandle_t handle, const RNNDescriptor& rnn_desc,
        const TensorDescriptor& x_desc, miopenDataType_t datatype)
{
    size_t weight_size;
    MIOPEN_CHECK(miopenGetRNNParamsSize(handle, rnn_desc.desc(), x_desc.desc(), &weight_size, datatype));
    auto element_size = dataSize(datatype);
    AT_ASSERTM(weight_size % element_size == 0, "miopenGetRNNParamsSize returned nonsensical weight_size.");
    return weight_size / element_size;
}

int64_t _num_linear_layers(miopenRNNMode_t mode) {
	switch(mode) {
		case miopenLSTM:
			return 8;
		case miopenGRU:
			return 6;
		case miopenRNNRELU:
			return 2;
		case miopenRNNTANH:
			return 2;
		default:
			AT_ERROR("Unknown miopen RNN mode : ", mode);
	}
}

std::pair<std::vector<Tensor>, size_t> get_parameters(miopenHandle_t handle, const RNNDescriptorParams& rnn,
					const RNNDescriptor& rnn_desc, const TensorDescriptor& x_desc, const FilterDescriptor& w_desc,
					const Tensor& weight_buf)
{
	/*TODO:
		1. implement _num_linear_layers method. [Done]
		2. Find equivalent to cudnnGetLinearMatrixParams and cudnnGetRNNLinearLayerBiasParams. (mioepnGetLayerParams, miopenGetBiasParams.) [Done]
		3. 
	*/
	std::vector<Tensor> params;
	int64_t num_linear_layers = _num_linear_layers(rnn.mode);
	int64_t num_layers = rnn.num_directions() * rnn.num_layers;
	size_t cur_offset = 0;
	size_t global_layer_params_count = 0;
	auto miopen_param_methods = { miopenGetRNNLayerParam, miopenGetRNNLayerBias};
	auto miopen_param_size_methods = {miopenGetRNNLayerParamSize, miopenGetRNNLayerBiasSize};

	for (int64_t layer=0; layer < num_layers; layer++) {
		size_t layer_params_count = 0;
		//Get all the weight parameters.
		for(int64_t linear_id = 0; linear_id < num_linear_layers; linear_id++) {
			FilterDescriptor lin_linear_mat_desc;
			void* matrix_pointer;
			size_t param_size;
			MIOPEN_CHECK(miopenGetRNNLayerParamSize(handle, rnn_desc.desc(), layer, x_desc.desc(), linear_id, &param_size));
			MIOPEN_CHECK(miopenGetRNNLayerParam(handle, rnn_desc.desc(), layer, x_desc.desc(), w_desc.desc(), weight_buf.data_ptr(), 
													lienar_id, lin_linear_mat_desc.mut_desc(), matrix_pointer));
			miopenDataType_t data_type;
			int nb_dims;
			constexpr int min_dim = 3;
			//Tensor filter_dim_a = at::empty(min_dim, at::InitialTensorOptions().dtype(kInt));
			//Tensor stride_dim_a = at::empty(min_dim, at::InitialTensorOptions().dtype(kInt));
			std::array<int,3> matDims {1,1,1};
			std::array<int,3> strideDims {1,1,1};
			MIOPEN_CHECK(miopenGetTensorDescriptor(lin_linear_mat_desc.desc(), &data_type, matDims.data(), strideDims.data()));

			auto elem_size = dataSize(getMiopenDataType(weight_buf));
			auto offset_bytes = (char *) matrix_pointer - (char *) weight_buf.data_ptr();
			AT_ASSERTM(offset_bytes % elem_size == 0, "offset_bytes = ", offset_bytes, "; elem_size = ", elem_size);
			size_t offset = offset_bytes / elem_size;

			int mat_numel = matDims[0] * matDims[1] * matDims[2];
			if (linear_id == 0 || linear_id == num_linear_layers / 2) {
				std::initializer_list<int64_t> size = {mat_numel * num_linear_layers / 2, 1};

				//Generate new parameter tensor which is a view into the weight_buf.
				Tensor param = at::empty({0}, weight_buf.options()).set_(weight_buf.storage(), offset, size);
				params.emplace_back(std::move(param));
				layer_params_count++;
			} else {
				AT_ASSERTM(cur_offset == offset, "cur_offset = ", cur_offset, " ; offset = ", offset);
			}

			cur_offset = offset + mat_numel;
		}

		//Get all the bias parameters.
		for(int64_t linear_id = 0; linear_id < num_linear_layers; linear_id++) {
			FilterDescriptor lin_linear_mat_desc;
			void* matrix_pointer;
			size_t param_size;
			MIOPEN_CHECK(miopenGetRNNLayerBiasSize(handle, rnn_desc.desc(), layer, x_desc.desc(), linear_id, &param_size));
			MIOPEN_CHECK(miopenGetRNNLayerBias(handle, rnn_desc.desc(), layer, x_desc.desc(), w_desc.desc(), weight_buf.data_ptr(), 
													lienar_id, lin_linear_mat_desc.mut_desc(), matrix_pointer));
			miopenDataType_t data_type;
			int nb_dims;
			constexpr int min_dim = 3;
			//Tensor filter_dim_a = at::empty(min_dim, at::InitialTensorOptions().dtype(kInt));
			//Tensor stride_dim_a = at::empty(min_dim, at::InitialTensorOptions().dtype(kInt));
			std::array<int,3> matDims {1,1,1};
			std::array<int,3> strideDims {1,1,1};
			MIOPEN_CHECK(miopenGetTensorDescriptor(lin_linear_mat_desc.desc(), &data_type, matDims.data(), strideDims.data()));

			auto elem_size = dataSize(getMiopenDataType(weight_buf));
			auto offset_bytes = (char *) matrix_pointer - (char *) weight_buf.data_ptr();
			AT_ASSERTM(offset_bytes % elem_size == 0, "offset_bytes = ", offset_bytes, "; elem_size = ", elem_size);
			size_t offset = offset_bytes / elem_size;

			int mat_numel = matDims[0] * matDims[1] * matDims[2];
			if (linear_id == 0 || linear_id == num_linear_layers / 2) {
				std::initializer_list<int64_t> size = {mat_numel * num_linear_layers / 2, 1};

				//Generate new parameter tensor which is a view into the weight_buf.
				Tensor param = at::empty({0}, weight_buf.options()).set_(weight_buf.storage(), offset, size);
				params.emplace_back(std::move(param));
				layer_params_count++;
			} else {
				AT_ASSERTM(cur_offset == offset, "cur_offset = ", cur_offset, " ; offset = ", offset);
			}

			cur_offset = offset + mat_numel;
		}

		if (layer == 0) {
			global_layer_params_count = layer_params_count;
		} else {
			 AT_ASSERTM(global_layer_params_count == layer_params_count,  "global_layer_params_count = ", global_layer_params_count, "; layer_params_count = ", layer_params_count);
		}
	}

	return std::make_pair(params, global_layer_params_count);
}

std::vector<int64_t> _input_size(const TensorDescriptorListParams& tensors) {
	if (tensors.is_input_packed()) {
		return {tensors.batch_sizes_sum, tensors.input_size};
	} else {
		return {tensors.seq_length, tensors.mini_batch, tensors.input_size};
	}
}

std::vector<int64_t> _hidden_size(const RNNDescriptorParams& rnn, const TensorDescriptorListParams& tensors) {
	return {rnn.num_layers * rnn.num_directions(), tensors.mini_batch, tensors.hidden_size};
}

std::vector<int64_t> _output_size(const RNNDescriptorParams& rnn, const TensorDescriptorListParams& tensors) {
	if (tensors.is_input_packed()) {
		return {tensors.batch_sizes_sum, rnn.hidden_size * rnn.num_directions()};
	} else {
		return {tensors.seq_length, tensors.mini_batch, rnn.hidden_size * rnn.num_directions()};
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

    RNNDescriptorParam rnn;
    rnn.set(fn_mode, hidden_size, num_layers, bidirectional, datatype);

    RNNDescriptor rnn_desc = rnn.descriptor();

    TensorGeometry x_geom({1, input_size});
    TensorDescriptor x_desc;
    x_desc.set(getMiopenDataType(any_param), x_geom.sizes(), x_geom.strides(), 5);

    auto num_weights = get_num_weights(handle, rnn_desc, x_desc, datatype);
    auto weight_buf = at::zeros(num_weights, any_param.options());

    FilterDescriptor w_desc;
    w_desc.set(weight_buf, 3);

    //Slice off views into weight_buf.
    std::vector<Tensor> params_arr;
    size_t params_stride0;
    std::tie(params_arr, params_stride0) = get_parameters(handle, rnn, rnn_desc, x_desc, w_desc, weight_buf);

    MatrixRef<Tensor> weight {weight_arr, static_cast<size_t>(weight_stride0)},
        params {params_arr, params_stride0};

    //Copy weights.
    _copyParams(weight, params);

    // Update the storage
    for (size_t i = 0; i < weight.size(0); i++) {
        for (auto orig_param_it = weight[i].begin(), new_param_it = params[i].begin();
                orig_param_it != weight[i].end() && new_param_it != params[i].end();
                orig_param_it++, new_param_it++) {
            auto orig_param = *orig_param_it, new_param = *new_param_it;
            orig_param.set_(new_param.view_as(orig_param));
        }
    }

    return weight_buf;

}

std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor> miopen_rnn(
        const Tensor& input_r, TensorList weight, int64_t weight_stride0,
        const Tensor& weight_buf_r, const Tensor& hx, const Tensor& cx,
        int64_t fn_mode, int64_t fn_hidden_size, int64_t fn_num_layers,
        bool batch_first, double fn_dropout, bool fn_train, bool fn_bidirectional,
        IntArrayRef fn_batch_sizes, const Tensor& fn_dropout_state
        ) {
    AT_ERROR("miopen_rnn : not implemented yet.");

    check_device(input_r, weight, {hx, cx});
    auto input = input_r;
    auto weight_buf = weight_buf_r;
    

    if (fn_dropout_state.defined()) {
    	AT_ERROR("miopen_rnn : Dropout is not supported in MIOpen. ");
    }

    RNNParams fn;
    auto datatype = getMiopenDataType(input);
    fn.rnn.set(fn_mode, fn_hidden_size, fn_num_layers, fn_bidirectional, datatype);
    fn.tensors.set(input.sizes(), fn_batch_sizes, batch_first);

    if (fn.rnn.mode != miopenLSTM) {
    	AT_CHECK(!cx.defined(), "miopen_rnn: illegal defined cx for non-LSTM RNN.");
    }

    auto is_input_packed = fn.tensors.batch_sizes.size() != 0;
    if (batch_first && !is_input_packed) {
    	input = input.transpose(0, 1);
    }

    auto hidden_size = _hidden_size(fn.rnn, fn.tensors);
    auto output_size = _output_size(fn.rnn, fn.tensors);

    AT_CHECK(hx.is_contiguous(), "miopen_rnn : hx is not contiguous.");
    AT_CHECK(!cx.defined() || cx.is_contiguous(), "miopen_rnn : cx is not contiguous.");

    auto x = input.contiguous();
    auto output = at::empty(output_size, input.options());
    auto hy = at::empty(hidden_size, hx.options());
    Tensor cy;
    if (cx.defined()) {
    	cy = at::empty(hidden_size, cx.options());
    } else {
    	cy = at::empty({0}, hx.options());
    }

    auto y = output;
    auto handle = getMiopenHandle();
    miopenRNNAlgo_t algo = miopenRNNdefault;
    fn.rnn.set_algo(algo);

    RNNDescriptors descs(fn, handle, x, y, hx, cx);

    //TODO: Need to implement get_parameters that gets params and params_stride0.
    FilterDescriptor w_desc;
    if (!weight_buf.defined()) {
    	auto num_weights = get_num_weights(handle, descs.rnn_desc(), descs.x_descs[0], datatype);
    	weight_buf = at::empty(num_weights, x.options());
    	w_desc.set(weight_buf, 3);
    	weight_buf.zero_();
    	std::vector<Tensor> params;
    	size_t params_stride0;
    	std::tie(params, params_stride0) = get_parameters(handle, fn.rnn, descs.rnn_desc, descs.x_descs[0], w_desc, weight_buf);
    	_copyParams(MatrixRef<Tensor>{weight, static_cast<size_t>(weight_stride0)},
    				MatrixRef<Tensor>{params, params_stride0});
    } else {
    	w_desc.set(weight_buf, 3);
    }

    AT_CHECK(!cx.defined() || cx.sizes().equals(hidden_size), "Expected cell size ", IntArrayRef{hidden_size}, ", got", cx.sizes());

    size_t workspace_size;
    auto x_descs_arr = descs.get_x_descs();
    auto y_descs_arr = descs.get_y_descs();

    //Allocate workspace size.
    MIOPEN_CHECK(miopenGetRNNWorkspaceSize(handle, descs.rnn_desc.desc(), fn.tensors.seq_length, x_descs_arr.data(), &workspace_size));
    auto workspace = at::empty(workspace_size, input.otpions().dtype(kByte));

    //Train or inference.
    Tensor reserve;
    if (fn_train) { //Train.
    	size_t reserver_size;
    	MIOPEN_CHECK(miopenGetRNNTrainingReserverSize(handle, descs.rnn_desc.desc(), fn.tensors.seq_length, x_descs_arr.data(), &reserver_size));
    	reserve = at::empty(reserver_size, input.options().dtype(kByte));

    	MIOPEN_CHECK(miopenRNNForwardTraining(handle, descs.rnn_desc.desc(), fn.tensors.seq_length,
    			x_descs_arr.data(), x.data_ptr(),
    			descs.hx_desc.desc(), hx.data_ptr(),
    			descs.cx_desc.desc(), cx.defined() ? cx.data_ptr() : nullptr,
    			w_desc.desc(), weight_buf.data_ptr(),
    			y_descs_arr.data(), y.data_ptr(),
    			descs.hy_desc.desc(), hy.data_ptr(),
    			descs.cy_desc.desc(), cy.defined() ? cy.data_ptr() : nullptr, 
    			workspace.data_ptr(), workspace_size, reserve.data_ptr(), reserver_size ));
    } else { //Inference.
    	reserve = at::empty({0}, input.options().dtype(kByte));
    	MIOPEN_CHECK(miopenRNNForwardInference(handle, descs.rnn_desc.desc(), fn.tensors.seq_length,
    			x_descs_arr.data(), x.data_ptr(),
    			descs.hx_desc.desc(), hx.data_ptr(),
    			descs.cx_desc.desc(), cx.defined() ? cx.data_ptr() : nullptr,
    			w_desc.desc(), weight_buf.data_ptr(),
    			y_descs_arr.data(), y.data_ptr(),
    			descs.hy_desc.desc(), hy.data_ptr(),
    			descs.cy_desc.desc(), cy.defined() ? cy.data_ptr() : nullptr,
    			workspace.data_ptr(), workspace_size));
    }

    if (batch_first && !is_input_packed) {
    	output.transpose_(0, 1);
    }

    return std::make_tuple(output, hy, cy, reserve, weight_buf);

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
