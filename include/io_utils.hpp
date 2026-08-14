// io_utils.hpp
// CSV loading, ADC conversion and artefact-path resolution.
#pragma once

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include <limits.h>
#include <unistd.h>
#include <sys/stat.h>

#include "extraction_config.hpp"
#include "feature_extraction.hpp"

inline void strip(std::string& s) {
    while (!s.empty() && (s.back() == '\r' || s.back() == ' ' || s.back() == '\n'))
        s.pop_back();
    size_t i = 0;
    while (i < s.size() && s[i] == ' ') i++;
    if (i) s.erase(0, i);
}

// Header-keyed, so columns this side does not know about are simply ignored —
// the logger can gain a channel without breaking the Pi.
inline std::map<std::string, std::vector<double>> load_csv_table(const std::string& path,
                                                                 size_t max_rows = 0) {
    std::map<std::string, std::vector<double>> table;
    std::ifstream file(path);
    if (!file) return table;

    std::string line;
    if (!std::getline(file, line)) return table;
    std::vector<std::string> cols;
    {
        std::stringstream ss(line);
        std::string col;
        while (std::getline(ss, col, ',')) { strip(col); cols.push_back(col); }
    }
    for (const auto& c : cols) table[c] = {};

    size_t rows = 0;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string val;
        size_t idx = 0;
        while (std::getline(ss, val, ',') && idx < cols.size()) {
            strip(val);
            double v = 0.0;
            if (!val.empty()) { try { v = std::stod(val); } catch (...) { v = 0.0; } }
            table[cols[idx]].push_back(v);
            idx++;
        }
        if (max_rows && ++rows >= max_rows) break;
    }
    return table;
}

// CSV columns -> internal roles, then raw ADC counts -> Amps / volts / g.
// Mirrors standardize_cols() + convert_units() in data_building.ipynb.
inline Channels to_channels(const std::map<std::string, std::vector<double>>& table,
                            const ExtractionConfig& c, bool already_physical) {
    Channels out;
    for (const auto& [csv_name, role] : c.column_map) {
        auto it = table.find(csv_name);
        if (it == table.end() || role == "t_raw") continue;
        out[role] = it->second;
    }
    if (already_physical || !c.raw_adc) return out;

    const AdcConfig& a = c.adc;
    for (const auto& role : c.current_roles) {
        auto it = out.find(role);
        if (it == out.end()) continue;
        double zero = a.current_midpoint;
        if (a.current_zero_mode == "file_median") {
            std::vector<double> v(it->second.size());
            for (size_t i = 0; i < v.size(); i++)
                v[i] = it->second[i] / a.adc_max * a.vref;
            std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
            zero = v[v.size() / 2];
        }
        for (double& x : it->second) {
            double volts = x / a.adc_max * a.vref;
            x = (volts - zero) / (a.shunt_r * a.csa_gain);
            x = std::max(-a.max_current, std::min(a.max_current, x));   // clip, as Python does
        }
    }
    for (const auto& role : c.voltage_roles) {
        auto it = out.find(role);
        if (it == out.end()) continue;
        for (double& x : it->second) x = x / a.adc_max * a.vref * a.volt_divider_gain;
    }
    if (out.count("speed_cmd"))
        for (double& x : out["speed_cmd"]) x = x / a.adc_max * a.vref * a.speed_cmd_gain;
    for (const auto& role : c.vib_roles) {
        auto it = out.find(role);
        if (it == out.end()) continue;
        for (double& x : it->second) x /= a.imu_sensitivity;
    }
    return out;
}

// data_building's load_file() drops the first SKIP_STARTUP_MS of a recording —
// the spin-up transient — but only when the file is long enough to spare it. A
// live window handed over by the service is short, so nothing is dropped there.
// The rule has to match Python exactly or the two sides window different samples.
inline size_t startup_skip(size_t n_rows, const ExtractionConfig& c,
                           double skip_ms) {
    if (skip_ms <= 0) return 0;
    size_t skip = static_cast<size_t>(skip_ms / 1000.0 * c.fs);
    return (n_rows > skip + static_cast<size_t>(c.window_samples)) ? skip : 0;
}

inline Channels slice(const Channels& src, size_t start, size_t count) {
    Channels out;
    for (const auto& [k, v] : src) {
        if (start >= v.size()) continue;
        size_t n = std::min(count, v.size() - start);
        out[k].assign(v.begin() + start, v.begin() + start + n);
    }
    return out;
}

inline bool is_dir(const std::string& p) {
    struct stat st;
    return stat(p.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

// Find the tree holding model/ and config/. Resolved from the executable rather
// than the working directory, because the service execs this binary from
// wherever systemd happens to leave it.
inline std::string find_root(const std::string& override_path) {
    if (!override_path.empty()) return override_path;

    char buf[PATH_MAX];
    ssize_t len = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    std::string dir = ".";
    if (len > 0) {
        buf[len] = '\0';
        std::string exe(buf);
        size_t slash = exe.find_last_of('/');
        if (slash != std::string::npos) dir = exe.substr(0, slash);
    }
    for (int up = 0; up < 4; up++) {
        if (is_dir(dir + "/model") && is_dir(dir + "/config")) return dir;
        dir += "/..";
    }
    return ".";
}
