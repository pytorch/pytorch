#pragma once

#include <string>
#include <iostream>
#include <../../../third_party/json/single_include/nlohmann/json.hpp>

namespace at {

struct DTRLogger {
  std::string time_prefix;
  std::ofstream out;
  static std::string get_time_prefix() {
    std::time_t t = std::time(nullptr);
    std::tm* tm = std::localtime(&t);
    return
      std::to_string(1900+tm->tm_year) + "-" +
      std::to_string(1+tm->tm_mon) + "-" +
      std::to_string(tm->tm_mday) + "-" +
      std::to_string(tm->tm_hour) + "-" +
      std::to_string(tm->tm_min) + "-" +
      std::to_string(tm->tm_sec);
  }
  std::string get_filename(const std::string& name) {
    return time_prefix + "-" + name + ".log";
  }
  DTRLogger() : time_prefix(get_time_prefix()), out(get_filename("default")) { }
  void log(const std::string& str) {
    out << str << std::endl;
  }
  static DTRLogger& logger() {
    static DTRLogger ret;
    return ret;
  }

};

using json = nlohmann::json;
constexpr bool log_json = true;
const std::string INSTRUCTION = "INSTRUCTION";
const std::string ANNOTATION = "ANNOTATION";
const std::string RELEASE = "RELEASE";
const std::string PIN = "PIN";
const std::string TIME = "TIME";
const std::string ARGS = "ARGS";
const std::string MEMORY = "MEMORY";
const std::string ALIAS = "ALIAS";
const std::string NAME = "NAME";
const std::string CONSTANT = "CONSTANT";

void DTRLogConstant(const std::string& name) {
  if (log_json) {
    json j;
    j[INSTRUCTION] = CONSTANT;
    j[NAME] = name;
    DTRLogger::logger().log(j.dump());
  } else {
    DTRLogger::logger().log(CONSTANT + " " + name);
  }
}

void DTRLogMemory(const std::string& name, size_t memory) {
  if (log_json) {
    json j;
    j[INSTRUCTION] = MEMORY;
    j[NAME] = name;
    j[MEMORY] = std::to_string(memory);
    DTRLogger::logger().log(j.dump());
  } else {
    DTRLogger::logger().log(name + " " + MEMORY + ": " + std::to_string(memory));
  }
}

void DTRLogAlias(const std::string& name, int index) {
  if (log_json) {
    json j;
    j[INSTRUCTION] = ALIAS;
    j[NAME] = name;
    j[ALIAS] = std::to_string(index);
    DTRLogger::logger().log(j.dump());
  } else {
    DTRLogger::logger().log(name + " " + ALIAS + ": " + std::to_string(index));
  }
}

void DTRLogCopyFrom(const std::string& to, const std::string& from) {
  if (log_json) {
    json j;
    j[INSTRUCTION] = "COPY_FROM";
    j["DST"] = to;
    j["SRC"] = from;
    DTRLogger::logger().log(j.dump());
  } else {
    DTRLogger::logger().log(to + " <- " + from);
  }
}

void DTRLogCopy(const std::string& new_name, const std::string& old_name) {
  if (log_json) {
    json j;
    j[INSTRUCTION] = "COPY";
    j["DST"] = new_name;
    j["SRC"] = old_name;
    DTRLogger::logger().log(j.dump());
  } else {
    DTRLogger::logger().log(new_name + " = " + old_name);
  }
}

void DTRLogMutate(const std::string& name,
                  const std::vector<std::string>& args,
                  const std::vector<size_t>& mutate,
                  const std::string& time) {
  if (log_json) {
    json j;
    j[INSTRUCTION] = "MUTATE";
    j[NAME] = name;
    j[ARGS] = args;
    j["MUTATE"] = mutate;
    j[TIME] = time;
    DTRLogger::logger().log(j.dump());
  } else {
    std::string log = name;
    log += "(";
    for (const auto& s : args) {
      log += s;
      log += ", ";
    }
    log += ") ";
    log += " MUTATING: ";
    log += "(";
    for (const size_t i : mutate) {
      log += std::to_string(i);
      log += ", ";
    }
    log += ") ";
    log += TIME;
    log += ": ";
    log += time;
    DTRLogger::logger().log(log);
  }
}

void DTRLogRelease(const std::string& name) {
  if (log_json) {
    json j;
    j[INSTRUCTION] = RELEASE;
    j[NAME] = name;
    DTRLogger::logger().log(j.dump());
  } else {
    DTRLogger::logger().log(RELEASE + ": " + name);
  }
}

void DTRLogPin(const std::string& name) {
  if (log_json) {
    json j;
    j[INSTRUCTION] = PIN;
    j[NAME] = name;
    DTRLogger::logger().log(j.dump());
  } else {
    DTRLogger::logger().log(RELEASE + ": " + name);
  }
}

void DTRLogCall(const std::vector<std::string>& res,
                const std::string& name,
                const std::vector<std::string>& args,
                const std::string& time) {
  if (log_json) {
    json j;
    j[INSTRUCTION] = "CALL";
    j[NAME] = name;
    j["RESULT"] = res;
    j[ARGS] = args;
    j[TIME] = time;
    DTRLogger::logger().log(j.dump());
  } else {
    std::string arg = name + "(";
    for (const auto& s : args) {
      arg += s;
      arg += ", ";
    }
    arg += ")";
    std::string log = "(";
    for (const auto& s: res) {
      log += s;
      log += ", ";
    }
    log += ") = ";
    log += arg;
    log += " TIME: ";
    log += time;
    DTRLogger::logger().log(log);
  }
}

}
