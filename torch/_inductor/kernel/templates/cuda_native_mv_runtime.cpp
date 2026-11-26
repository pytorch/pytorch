struct CudaNativeMvContext {
  c10::cuda::CUDAStreamGuard stream_guard;
  at::Tensor source_matrix, source_vector, matrix, vector, partial;
  int64_t M, K, S0, S1, VS, CTAS = 1, THREADS = 1, ACTIVE_THREADS = 1;
  int64_t INNER_CHUNK = 1, INNER_STRIDE = 1, KIND = 0, K_SLICE = 0;
  int64_t SWAP = 0, PRODUCT = 0, PRE_SCALE = 0;
  bool copy_matrix = false, copy_vector = false;

  CudaNativeMvContext(AtenTensorHandle x, AtenTensorHandle v, cudaStream_t stream)
      : stream_guard(c10::cuda::getStreamFromExternal(
            stream, torch::aot_inductor::tensor_handle_to_tensor_pointer(x)->get_device())),
        source_matrix(*torch::aot_inductor::tensor_handle_to_tensor_pointer(x)),
        source_vector(*torch::aot_inductor::tensor_handle_to_tensor_pointer(v)),
        matrix(source_matrix), vector(source_vector),
        M(matrix.size(0)), K(matrix.size(1)),
        S0(matrix.stride(0)), S1(matrix.stride(1)), VS(vector.stride(0)) {
    if (M == 0 || K == 0) return;
    const bool native_matrix = (S0 == 1 && S1 >= std::max<int64_t>(1, M)) ||
        (S1 == 1 && S0 >= std::max<int64_t>(1, K));
    copy_matrix = !native_matrix || matrix.is_neg() || matrix.is_conj();
    copy_vector = VS == 0 || vector.is_neg() || vector.is_conj();
    if (copy_matrix) {
      matrix = native_matrix && matrix.is_non_overlapping_and_dense()
          ? at::empty_strided(matrix.sizes(), matrix.strides(), matrix.options())
          : at::empty(matrix.sizes(), matrix.options());
    }
    if (copy_vector) {
      vector = VS != 0 && vector.is_non_overlapping_and_dense()
          ? at::empty_strided(vector.sizes(), vector.strides(), vector.options())
          : at::empty(vector.sizes(), vector.options());
    }
    S0 = matrix.stride(0);
    S1 = matrix.stride(1);
    VS = vector.stride(0);
    const bool transposed = !(S0 == 1 && S1 >= std::max<int64_t>(1, M));
    auto plan = at::cuda::blas::getGemvLaunchPlan(
        matrix.scalar_type(), transposed ? K : M, transposed ? M : K,
        transposed ? S0 : S1, VS, transposed,
        reinterpret_cast<uintptr_t>(matrix.data_ptr()) & 15,
        reinterpret_cast<uintptr_t>(vector.data_ptr()) & 15);
    auto primary = std::find_if(plan.begin(), plan.end(), [](const auto& launch) {
      return std::get<0>(launch).find("reduce") == std::string::npos;
    });
    AOTI_TORCH_CHECK(primary != plan.end(), "Missing native GEMV launch");
    const auto& name = std::get<0>(*primary);
    const auto& grid = std::get<1>(*primary);
    const auto& block = std::get<2>(*primary);
    auto partition = std::get<4>(*primary);
    const bool split = std::any_of(plan.begin(), plan.end(), [](const auto& launch) {
      return std::get<0>(launch).find("splitKreduce") != std::string::npos;
    });
    int64_t tile_k = 1;
    if (name.find("dot_kernel") != std::string::npos) {
      KIND = 2; CTAS = grid[0]; ACTIVE_THREADS = block[0];
    } else if (name.rfind("nvjet_", 0) == 0) {
      std::smatch tiles;
      AOTI_TORCH_CHECK(std::regex_search(name, tiles,
          std::regex("nvjet_[^_]+_(\\d+)x(\\d+)_(\\d+)x(\\d+)_")),
          "Unknown native GEMV tensor-core geometry");
      const int64_t tile_m = std::max(std::stoll(tiles[1]), std::stoll(tiles[2]));
      tile_k = std::stoll(tiles[3]);
      KIND = split ? 5 : 4;
      if (split) CTAS = grid[0] * grid[1] * grid[2] / ((M + tile_m - 1) / tile_m);
    } else if (split) {
      KIND = 3; CTAS = grid[2]; ACTIVE_THREADS = block[1];
    } else if (name.find("gemvx") != std::string::npos) {
      std::regex flags("Lb([01])E");
      std::vector<int> values;
      for (auto i = std::sregex_iterator(name.begin(), name.end(), flags);
           i != std::sregex_iterator(); ++i) values.push_back(std::stoi((*i)[1]));
      AOTI_TORCH_CHECK(values.size() >= 3, "Missing native GEMV layout flags");
      ACTIVE_THREADS = values[2] ? block[0] : block[1];
    } else if (name.find("gemv2N_kernel") != std::string::npos) {
      ACTIVE_THREADS = 1;
    } else if (name.find("gemvNSP_kernel") != std::string::npos) {
      ACTIVE_THREADS = block[1]; partition = {4, 1, 0};
    } else {
      AOTI_TORCH_CHECK(K == 1, "Unknown native GEMV reduction geometry");
    }
    while (THREADS < ACTIVE_THREADS) THREADS *= 2;
    INNER_CHUNK = partition[0]; INNER_STRIDE = partition[1];
    SWAP = KIND == 2 ? !partition[2] : partition[2];
    PRODUCT = name.find("gemmk1_kernel") != std::string::npos;
    PRE_SCALE = name.find("gemv2N_kernel") != std::string::npos;
    if (KIND == 4 || KIND == 5) {
      K_SLICE = ((K + CTAS - 1) / CTAS + tile_k - 1) / tile_k * tile_k;
    }
    const int64_t components = matrix.is_complex() ? 2 : 1;
    const auto acc = matrix.scalar_type() == at::kDouble ||
        matrix.scalar_type() == at::kComplexDouble ? at::kDouble : at::kFloat;
    partial = at::empty({M, CTAS, components}, matrix.options().dtype(acc));
  }
};
