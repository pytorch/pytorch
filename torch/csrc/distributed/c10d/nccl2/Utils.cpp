// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <torch/csrc/distributed/c10d/nccl2/Utils.hpp>
#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>

namespace c10d::nccl2 {

namespace {
// Helper function to trim leading and trailing whitespace from a string
std::string trim_whitespace(std::string_view str) {
  auto start = str.find_first_not_of(" \t\n\r\f\v");
  if (start == std::string_view::npos) {
    return "";
  }
  auto end = str.find_last_not_of(" \t\n\r\f\v");
  return std::string(str.substr(start, end - start + 1));
}

// Note: PALS does not provide an env variable for size, like `PALS_SIZE`.
// We just check it if supplied in the future. We try to query size from other
// env vars that may be set in a PALS environment, such as PMI_SIZE, WORLD_SIZE,
// etc.
// TODO: replace with the correct PALS env var for size once it is available.
int query_pals_size() {
  int size = env_to_value<int>("PALS_SIZE", -1);
  if (size > 0) {
    return size;
  }

  // MPICH sets PMI_SIZE in its environment.
  size = env_to_value<int>("PMI_SIZE", -1);
  if (size > 0) {
    return size;
  }

  // OpenMPI sets OMPI_COMM_WORLD_SIZE in its environment.
  size = env_to_value<int>("OMPI_COMM_WORLD_SIZE", -1);
  if (size > 0) {
    return size;
  }

  // Try WORLD_SIZE as a last resort
  size = env_to_value<int>("WORLD_SIZE", -1);
  if (size > 0) {
    return size;
  }

  return -1;
}

} // namespace

bool string_to_bool(std::string_view str) {
  std::string lowercase_str = trim_whitespace(str);
  std::transform(
      lowercase_str.begin(),
      lowercase_str.end(),
      lowercase_str.begin(),
      [](unsigned char c) { return std::tolower(c); });

  bool is_true =
      (lowercase_str == "1" || lowercase_str == "true" ||
       lowercase_str == "yes" || lowercase_str == "y");
  bool is_false =
      (lowercase_str == "0" || lowercase_str == "false" ||
       lowercase_str == "no" || lowercase_str == "n");

  if (!is_true && !is_false) {
    throw std::runtime_error("Invalid value for string " + std::string(str));
  } else {
    return is_true;
  }
}

template <typename T>
T env_to_value(std::string_view env_key, const T& default_value) {
  const char* env_value = std::getenv(std::string(env_key).c_str());
  if (!env_value) {
    return default_value; // Environment variable not set, return default
  }

  std::string value = trim_whitespace(std::string(env_value));

  // If the trimmed value is empty, return the default
  if (value.empty()) {
    return default_value;
  }

  if constexpr (std::is_same_v<T, bool>) {
    return string_to_bool(value);
  } else if constexpr (std::is_same_v<T, std::string>) {
    // String conversion (just return the value as is; NRVO, no std::move)
    return value;
  } else {
    // For all other types, use stringstream for conversion
    try {
      T result;
      std::istringstream ss(value);
      ss >> result;

      if (ss.fail() || !ss.eof()) {
        throw std::runtime_error("Conversion failed");
      }

      return result;
    } catch (const std::exception&) {
      throw std::runtime_error(
          "Invalid value for environment variable " + std::string(env_key) +
          ": " + value);
    }
  }
}

// Explicit instantiations for common types
template bool env_to_value<bool>(std::string_view, const bool&);
template int env_to_value<int>(std::string_view, const int&);
template float env_to_value<float>(std::string_view, const float&);
template double env_to_value<double>(std::string_view, const double&);
template uint64_t env_to_value<uint64_t>(std::string_view, const uint64_t&);
template std::string env_to_value<std::string>(
    std::string_view,
    const std::string&);

std::pair<int, int> query_ranksize() {
  // Constants for ranksize query methods
  const std::string kRanksizeQueryMethodAuto = "auto";
  const std::string kRanksizeQueryMethodTorchrun = "torchrun";
  const std::string kRanksizeQueryMethodMPI = "mpi";
  const std::string kRanksizeQueryMethodPALS = "pals";
  const std::string& kRanksizeQueryMethodDefault = kRanksizeQueryMethodAuto;

  // Get the ranksize query method from environment variable
  const char* ranksize_env =
      std::getenv("TORCHCOMM_BOOTSTRAP_RANKSIZE_QUERY_METHOD");
  std::string ranksize_query_method;
  if (ranksize_env == nullptr) {
    ranksize_query_method = kRanksizeQueryMethodDefault;
  } else {
    ranksize_query_method = ranksize_env;
  }

  // Convert to lowercase for comparison
  std::transform(
      ranksize_query_method.begin(),
      ranksize_query_method.end(),
      ranksize_query_method.begin(),
      [](unsigned char c) { return std::tolower(c); });

  int rank;
  int comm_size;

  // Lambda to query rank and size from environment variables, returning true if
  // both values were found
  auto tryQueryRankSize = [&]() -> bool {
    // Read from TORCHCOMM_RANK and TORCHCOMM_SIZE environment variables
    rank = env_to_value<int>("TORCHCOMM_RANK", -1);
    comm_size = env_to_value<int>("TORCHCOMM_SIZE", -1);
    if (rank > -1 && comm_size > 0) {
      return true;
    }

    // See if we are in an MPI environment
    if (ranksize_query_method == kRanksizeQueryMethodAuto ||
        ranksize_query_method == kRanksizeQueryMethodMPI) {
      // See if we are in an OpenMPI environment
      rank = env_to_value<int>("OMPI_COMM_WORLD_RANK", -1);
      comm_size = env_to_value<int>("OMPI_COMM_WORLD_SIZE", -1);
      if (rank > -1 && comm_size > 0) {
        return true;
      }

      // See if we are in an MPICH environment
      rank = env_to_value<int>("PMI_RANK", -1);
      comm_size = env_to_value<int>("PMI_SIZE", -1);
      if (rank > -1 && comm_size > 0) {
        return true;
      }
    }

    // See if we are in a PALS environment
    if (ranksize_query_method == kRanksizeQueryMethodAuto ||
        ranksize_query_method == kRanksizeQueryMethodPALS) {
      rank = env_to_value<int>("PALS_RANKID", -1);
      comm_size = query_pals_size();
      if (rank > -1 && comm_size > 0) {
        return true;
      }
    }

    // See if we are in a Torchrun environment
    if (ranksize_query_method == kRanksizeQueryMethodAuto ||
        ranksize_query_method == kRanksizeQueryMethodTorchrun) {
      rank = env_to_value<int>("RANK", -1);
      comm_size = env_to_value<int>("WORLD_SIZE", -1);
      if (rank > -1 && comm_size > 0) {
        return true;
      }
    }

    return false;
  };

  if (!tryQueryRankSize()) {
    throw std::runtime_error(
        "Unable to determine rank and size from environment variables. "
        "Please set TORCHCOMM_RANK and TORCHCOMM_SIZE, or ensure you are "
        "running in a supported environment (Torchrun, MPI, or PALS).");
  }

  return std::make_pair(rank, comm_size);
}

} // namespace c10d::nccl2
