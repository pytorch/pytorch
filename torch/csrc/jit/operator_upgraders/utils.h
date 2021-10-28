#pragma once
#include <iostream>
#include <string>
#include <regex>
#include <vector>
#include <torch/csrc/jit/operator_upgraders/version_map.h>

namespace torch {
namespace jit {

static UpgraderEntry findUpgrader(std::vector<UpgraderEntry> upgraders_for_schema, int current_version) {
    // we want to find the entry which satisfies following two conditions:
    //    1. the version entry must be greater than current_version
    //    2. Among the version entries, we need to see if the current version
    //       is in the upgrader name range
    std::cout << current_version << std::endl;
    for (const auto& upgrader_entry: upgraders_for_schema) {
        if (upgrader_entry.version_bump > current_version) {
            auto upgrader_name = upgrader_entry.upgrader_name;
            std::regex delimiter("_");
            std::vector<std::string> tokens(std::sregex_token_iterator(upgrader_name.begin(), upgrader_name.end(), delimiter, -1),
                                  std::sregex_token_iterator());

            int start = std::stoi(tokens[tokens.size()-2]);
            int end = std::stoi(tokens[tokens.size()-1]);

            if (start <= current_version && current_version <= end) {
                std::cout << "HERE\n";
                return upgrader_entry;
            }

        }
    }
    return {};
}

} // namespace jit
} // namespace torch
