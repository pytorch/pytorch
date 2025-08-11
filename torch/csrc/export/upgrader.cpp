#include <torch/csrc/export/upgrader.h>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace torch::_export {

// Global upgrader registry organized by version.
// Using std::multiset to maintain automatic bottom-up ordering where
// deeper keypaths are processed before shallower ones.
static std::map<int, std::multiset<Upgrader>> upgrader_registry;

static const std::multiset<Upgrader>& getUpgrader(int current_version) {
  static const std::multiset<Upgrader> empty_upgraders;
  auto it = upgrader_registry.find(current_version);
  if (it != upgrader_registry.end()) {
    return it->second;
  }
  return empty_upgraders;
}

static nlohmann::json getFieldByKeypath(
    const nlohmann::json& obj,
    const std::vector<std::string>& keypath) {
  nlohmann::json current = obj;
  for (const auto& key : keypath) {
    if (!current.contains(key)) {
      throw std::runtime_error("Keypath not found: " + key);
    }
    current = current[key];
  }
  return current;
}

static void setFieldByKeypath(
    nlohmann::json& obj,
    const std::vector<std::string>& keypath,
    const nlohmann::json& value) {
  nlohmann::json* current = &obj;
  for (size_t i = 0; i < keypath.size() - 1; ++i) {
    const auto& key = keypath[i];
    if (!current->contains(key)) {
      throw std::runtime_error("Keypath not found: " + key);
    }
    current = &((*current)[key]);
  }
  if (!current->contains(keypath.back())) {
    throw std::runtime_error("Keypath not found: " + keypath.back());
  }
  (*current)[keypath.back()] = value;
}

Upgrader::Upgrader(std::vector<std::string> kp, UpgraderFunction func)
    : keypath(std::move(kp)), upgrade_func(std::move(func)) {}

bool Upgrader::operator<(const Upgrader& other) const {
  // First compare by depth - deeper paths come first for bottom-up processing
  if (keypath.size() != other.keypath.size()) {
    return keypath.size() > other.keypath.size();
  }
  // If same depth, compare lexicographically for deterministic ordering
  return keypath < other.keypath;
}

void registerUpgrader(
    int version,
    const std::vector<std::string>& keypath,
    const UpgraderFunction& upgrade_func) {
  // Check if an upgrader already exists for this version and keypath
  auto version_it = upgrader_registry.find(version);
  if (version_it != upgrader_registry.end()) {
    const auto& upgraders = version_it->second;

    // Search for existing upgrader with the same keypath
    for (const auto& existing_upgrader : upgraders) {
      if (existing_upgrader.keypath == keypath) {
        std::ostringstream error_stream;
        error_stream << "Upgrader already registered for version " << version
                     << " and keypath: ";
        for (size_t i = 0; i < keypath.size(); ++i) {
          if (i > 0)
            error_stream << ".";
          error_stream << keypath[i];
        }
        throw std::runtime_error(error_stream.str());
      }
    }
  }

  upgrader_registry[version].emplace(keypath, upgrade_func);
}

void registerUpgrader(
    int version,
    const std::string& dot_keypath,
    const UpgraderFunction& upgrade_func) {
  // Convert dot-separated keypath to vector and delegate to main implementation
  std::vector<std::string> keypath_vector;
  std::stringstream ss(dot_keypath);
  std::string component;

  while (std::getline(ss, component, '.')) {
    if (component.empty()) {
      throw std::invalid_argument("Empty component in keypath: " + dot_keypath);
    }
    keypath_vector.push_back(component);
  }

  if (keypath_vector.empty()) {
    throw std::invalid_argument("Empty keypath provided");
  }

  registerUpgrader(version, keypath_vector, upgrade_func);
}

bool deregisterUpgrader(int version, const std::vector<std::string>& keypath) {
  auto version_it = upgrader_registry.find(version);
  if (version_it == upgrader_registry.end()) {
    return false; // Version not found
  }

  auto& upgraders = version_it->second;

  // Find the upgrader with matching keypath
  for (auto it = upgraders.begin(); it != upgraders.end(); ++it) {
    if (it->keypath == keypath) {
      upgraders.erase(it);

      // If this was the last upgrader for this version, remove the version
      // entry
      if (upgraders.empty()) {
        upgrader_registry.erase(version_it);
      }

      return true; // Successfully removed
    }
  }

  return false; // Upgrader not found
}

bool deregisterUpgrader(int version, const std::string& dot_keypath) {
  // Convert dot-separated keypath to vector and delegate to main implementation
  std::vector<std::string> keypath_vector;
  std::stringstream ss(dot_keypath);
  std::string component;

  while (std::getline(ss, component, '.')) {
    if (component.empty()) {
      throw std::invalid_argument("Empty component in keypath: " + dot_keypath);
    }
    keypath_vector.push_back(component);
  }

  if (keypath_vector.empty()) {
    throw std::invalid_argument("Empty keypath provided");
  }

  return deregisterUpgrader(version, keypath_vector);
}

void throwUpgraderError(
    const std::string& upgrader_name,
    int from_version,
    const std::string& error_message,
    const nlohmann::json& problematic_object) {
  std::ostringstream error_stream;
  error_stream << "Error in upgrader '" << upgrader_name << "' "
               << "while upgrading from version " << from_version
               << " to version " << from_version + 1 << ": " << error_message;

  if (!problematic_object.empty()) {
    error_stream << "\nProblematic object: " << problematic_object.dump(2);
  }

  throw std::runtime_error(error_stream.str());
}

nlohmann::json upgrade(const nlohmann::json& artifact, int target_version) {
  auto current_artifact = artifact;

  // Validate that the artifact contains required schema version information
  if (!current_artifact.contains("schema_version")) {
    throw std::runtime_error("Missing schema_version field in artifact");
  }

  int current_version = current_artifact["schema_version"]["major"];

  // Iteratively apply upgraders until target version is reached or no more are
  // available
  while (current_version < target_version) {
    // Look up upgraders for the current version
    const auto& upgraders = getUpgrader(current_version);

    if (upgraders.empty()) {
      // No more upgraders available - stop upgrading
      break;
    }

    // Apply all upgraders for this version in bottom-up order
    // (deeper keypaths first to prevent parent/child conflicts)
    for (const auto& upgrader : upgraders) {
      // Extract the field to be upgraded using its keypath
      auto field_to_upgrade =
          getFieldByKeypath(current_artifact, upgrader.keypath);

      // Apply the upgrade transformation
      auto upgraded_field = upgrader.upgrade_func(field_to_upgrade);

      // Update the artifact with the upgraded field
      setFieldByKeypath(current_artifact, upgrader.keypath, upgraded_field);
    }

    // Move to the next version for potential additional upgrades
    current_version++;
  }

  // Update schema version to reflect the final upgraded version
  if (current_artifact["schema_version"]["major"] != current_version) {
    current_artifact["schema_version"]["major"] = current_version;
    // Reset minor version to 0 - the correct minor version should be set
    // when converting the json to in memory representation of ExportedProgram
    current_artifact["schema_version"]["minor"] = 0;
  }

  // Validate that we reached the target version if requested
  if (current_version != target_version) {
    std::ostringstream error_stream;
    error_stream
        << "Failed to upgrade to target version " << target_version
        << ". Final version reached: " << current_version
        << ". This may indicate missing upgraders for intermediate versions.";
    throw std::runtime_error(error_stream.str());
  }

  return current_artifact;
}

} // namespace torch::_export
