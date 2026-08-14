// extraction_config.hpp
// Reads config/feature_extraction.json — the contract written by
// host_pipeline/data_building.ipynb.
//
// Every constant in here also exists in the Python that trained the models.
// In the previous C++ they were hard-coded in both languages and drifted: the
// old adc_conversion.hpp still carries a warning that its amplifier gain might
// not match the Python config. Loading them from one file removes that.
#pragma once

#include <map>
#include <string>
#include <vector>
#include <fstream>
#include <stdexcept>

#include "nlohmann/json.hpp"

struct AdcConfig {
    double adc_max = 4095.0;
    double vref = 3.3;
    double shunt_r = 0.00375;
    double csa_gain = 50.0;
    double current_midpoint = 1.65;
    double volt_divider_gain = 1.0;
    double speed_cmd_gain = 1.0;
    double imu_sensitivity = 16384.0;
    double max_current = 17.6;
    std::string current_zero_mode = "config";
};

struct ExtractionConfig {
    std::string data_source;
    double fs = 20000.0, fs_vib = 20000.0;
    int window_samples = 1000, context_samples = 4000;
    int step = 0;                                    // 0 -> window_samples (no overlap)
    double window_ms = 50.0, context_ms = 200.0;
    std::string vib_mode = "same_rate";
    bool raw_adc = true;

    std::map<std::string, std::string> column_map;   // csv column -> internal role
    AdcConfig adc;

    int pole_pairs = 13;
    double max_rpm = 740.0;
    double rpm_lo = 5.0, rpm_hi = 192.4;

    std::vector<std::string> time_channels, freq_channels;
    std::vector<std::string> current_roles, voltage_roles, vib_roles;
    std::string mcsa_channel = "I1", order_channel = "az";
    std::vector<std::pair<double, double>> bands;

    bool ctx_rpm = true, ctx_mcsa = true, ctx_ratios = true, ctx_order = true;
    std::map<std::string, bool> groups, time_features, freq_features;
    double eps = 1e-9;
    std::vector<std::string> feature_schema;
    double skip_startup_ms = 0.0;
};

inline ExtractionConfig load_extraction_config(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open extraction config: " + path);
    nlohmann::json j;
    in >> j;

    ExtractionConfig c;
    c.data_source = j.value("data_source", "");
    c.fs = j.at("fs").get<double>();
    c.fs_vib = j.value("fs_vib", c.fs);
    c.window_samples = j.at("window_samples").get<int>();
    c.context_samples = j.value("context_samples", 0);
    c.step = j.value("step", c.window_samples);
    c.window_ms = j.value("window_ms", 0.0);
    c.context_ms = j.value("context_ms", 0.0);
    c.vib_mode = j.value("vib_mode", "same_rate");
    c.raw_adc = j.value("raw_adc", true);
    c.eps = j.value("eps", 1e-9);
    c.skip_startup_ms = j.value("skip_startup_ms", 0.0);

    for (auto& [k, v] : j.at("column_map").items())
        c.column_map[k] = v.get<std::string>();

    const auto& a = j.at("adc");
    c.adc.adc_max = a.value("adc_max", 4095.0);
    c.adc.vref = a.value("vref", 3.3);
    c.adc.shunt_r = a.value("shunt_r", 0.00375);
    c.adc.csa_gain = a.value("csa_gain", 50.0);
    c.adc.current_midpoint = a.value("current_midpoint", c.adc.vref / 2);
    c.adc.volt_divider_gain = a.value("volt_divider_gain", 1.0);
    c.adc.speed_cmd_gain = a.value("speed_cmd_gain", 1.0);
    c.adc.imu_sensitivity = a.value("imu_sensitivity", 16384.0);
    c.adc.max_current = a.value("max_current", 17.6);
    c.adc.current_zero_mode = a.value("current_zero_mode", "config");

    const auto& m = j.at("motor");
    c.pole_pairs = m.value("pole_pairs", 13);
    c.max_rpm = m.value("max_rpm", 740.0);
    if (m.contains("rpm_search_hz")) {
        c.rpm_lo = m["rpm_search_hz"][0].get<double>();
        c.rpm_hi = m["rpm_search_hz"][1].get<double>();
    }

    const auto& ch = j.at("channels");
    c.time_channels = ch.at("time").get<std::vector<std::string>>();
    c.freq_channels = ch.at("freq").get<std::vector<std::string>>();
    c.current_roles = ch.value("current", std::vector<std::string>{"I1", "I2", "I3"});
    c.voltage_roles = ch.value("voltage", std::vector<std::string>{"V1", "V2", "V3", "vdc"});
    c.vib_roles = ch.value("vibration", std::vector<std::string>{"ax", "ay", "az"});
    c.mcsa_channel = ch.value("mcsa", "I1");
    c.order_channel = ch.value("order", "az");

    for (const auto& b : j.at("freq_bands"))
        c.bands.emplace_back(b[0].get<double>(), b[1].get<double>());

    const auto& cf = j.at("context_features");
    c.ctx_rpm = cf.value("rpm", true);
    c.ctx_mcsa = cf.value("mcsa", true);
    c.ctx_ratios = cf.value("ratios", true);
    c.ctx_order = cf.value("order", true);

    for (auto& [k, v] : j.at("feature_groups").items()) c.groups[k] = v.get<bool>();
    for (auto& [k, v] : j.at("time_features").items()) c.time_features[k] = v.get<bool>();
    for (auto& [k, v] : j.at("frequency_features").items()) c.freq_features[k] = v.get<bool>();
    c.feature_schema = j.value("feature_schema", std::vector<std::string>{});
    return c;
}
