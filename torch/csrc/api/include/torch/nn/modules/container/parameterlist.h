#pragma once
#include <iostream>

#include <torch/nn/cloneable.h>
#include <torch/torch.h>
#include <ATen/Tensor.h>

#include <vector>

namespace torch {
	namespace nn {

/// A list of Parameter 


		class ParameterListImpl : public Cloneable<ParameterListImpl> {
		public:
			using Iterator = std::vector<torch::Tensor>::iterator;
			using ConstIterator = std::vector<torch::Tensor>::const_iterator;

			ParameterListImpl() = default;
			template <typename... Tensors>
			explicit ParameterListImpl(Tensors&&... params) {
				params_.reserve(sizeof...(Tensors));
				push_back_var(std::forward<Tensors>(params)...);
			}

			template <typename... Tensors>
			explicit ParameterListImpl(const Tensors&... params) {
				params_.reserve(sizeof...(Tensors));
				push_back_var(std::forward<Tensors>(params)...);
			}

			void push_back(torch::Tensor &&param) {
				params_.push_back(std::move(param));
				const auto idx = params_.size() - 1;
				register_parameter(c10::to_string(idx), params_[idx]);
			}

			void push_back(const torch::Tensor &param) {
				params_.push_back(param);
				const auto idx = params_.size() - 1;
				register_parameter(c10::to_string(idx), params_[idx]);
			}


			/// Append each parameter in the container to the end of the list 
			template <typename T>
			void extend(const T& container) {
				for (const auto& param : container) {
					push_back(param);
				}
			}


			/// `reset()` is empty for `ParameterList`, since it does not have parameters of
			/// its own.
			void reset() override {}

			/// Pretty prints the `ParameterList` module into the given `stream`.
			void pretty_print(std::ostream& stream) const override {
				stream << "ParameterList(" << std::endl;
				for (const auto& para : params_) {
					stream << std::endl;
					stream << para;
				}
				stream << ")";
			}

			/// Returns an iterator to the start of the ParameterList
			Iterator begin() {
				return params_.begin();
			}

			/// Returns a const iterator to the start of the ParameterList
			ConstIterator begin() const {
				return params_.begin();
			}

			/// Returns an iterator to the end of the ParameterList
			Iterator end() {
				return params_.end();
			}

			/// Returns a const iterator to the end of the ParameterList
			ConstIterator end() const {
				return params_.end();
			}

			//Return the parameter by the given index
			at::Tensor& at(size_t idx) {
				return params_[idx];
			}

			const at::Tensor& at(size_t idx) const {
				return params_[idx];
			}



			const at::Tensor& operator [](size_t idx) const {
				return at(idx);
			}

			/// Return the size of the ParameterList
			size_t size() const noexcept {
				return params_.size();
			}
			/// True if the ParameterList is empty
			bool is_empty() const noexcept {
				return params_.empty();
			}
		private:
			std::vector<torch::Tensor> params_;

			/// Move constructor call 

			template <typename Head, typename... Tail>
			void push_back_var(Head&& head, Tail&&... tail) {
				push_back(std::forward<Head>(head));
				// Recursively calls this method, until the parameter pack only thas this
				// entry left. Then calls `push_back()` a final time (above).
				push_back_var(std::forward<Tail>(tail)...);
			}

			///Copy constructor call 

			template <typename Head, typename... Tail>
			void push_back_var(const Head& head, const Tail&... tail) {
				push_back(std::forward<Head>(head));
				// Recursively calls this method, until the parameter pack only thas this
				// entry left. Then calls `push_back()` a final time (above).
				push_back_var(std::forward<Tail>(tail)...);
			}

			/// The base case, when the list of modules is empty.
			void push_back_var() {}

		};
		TORCH_MODULE(ParameterList);
	} //namespace nn
} //namespace torch