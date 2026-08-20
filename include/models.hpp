// models.hpp
// The three models, each loaded from JSON and evaluated with plain arithmetic.
//
// Nothing here links TensorFlow. The classifier is a ~3k-parameter CNN whose
// weights are exported as JSON, the anomaly detector is a mean vector and a
// precision matrix, and the RUL model is a linear map plus a lookup table. The
// previous C++ needed a hand-built TFLite in a developer's home directory to
// link at all; this builds with cmake and a compiler.
//
// model.tflite is still exported by the notebook for anyone who wants that path.
#pragma once

#include <algorithm>
#include <cmath>
#include <fstream>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"

// ── shared: feature order + scaler ────────────────────────────────────
struct TaskInput {
    std::vector<std::string> feature_order;
    std::vector<double> mean, scale;

    void load(const std::string& config_dir) {
        nlohmann::json order, sc;
        std::ifstream(config_dir + "/feature_order.json") >> order;
        std::ifstream(config_dir + "/scaler.json") >> sc;
        feature_order = order.get<std::vector<std::string>>();
        mean = sc.at("mean").get<std::vector<double>>();
        scale = sc.at("scale").get<std::vector<double>>();
        if (feature_order.size() != mean.size() || mean.size() != scale.size())
            throw std::runtime_error(config_dir +
                ": feature_order, scaler mean and scale disagree in length");
    }

    // Build the model's input vector by NAME. Order is the contract: the same
    // names in the same order the model was trained on.
    std::vector<double> vectorise(const std::map<std::string, double>& features,
                                  std::vector<std::string>* missing = nullptr) const {
        std::vector<double> z(feature_order.size(), 0.0);
        for (size_t i = 0; i < feature_order.size(); i++) {
            auto it = features.find(feature_order[i]);
            double raw = 0.0;
            if (it == features.end()) {
                if (missing) missing->push_back(feature_order[i]);
            } else {
                raw = it->second;
            }
            double s = scale[i] == 0.0 ? 1.0 : scale[i];
            z[i] = (raw - mean[i]) / s;
        }
        return z;
    }
};

// ── classification: sequential 1D CNN from JSON weights ───────────────
struct CnnLayer {
    std::string type, activation, padding;
    int kernel_size = 0, filters = 0, pool_size = 0, strides = 0, units = 0;
    std::vector<std::vector<std::vector<double>>> kernel3;  // conv: [k][in][out]
    std::vector<std::vector<double>> kernel2;               // dense: [in][out]
    std::vector<double> bias;
};

struct Classifier {
    std::vector<CnnLayer> layers;
    std::vector<std::string> classes;
    int input_length = 0;

    void load(const std::string& model_dir) {
        nlohmann::json j;
        std::ifstream(model_dir + "/model_weights.json") >> j;
        input_length = j.at("input_length").get<int>();
        classes = j.at("classes").get<std::vector<std::string>>();
        for (const auto& l : j.at("layers")) {
            CnnLayer L;
            L.type = l.at("type").get<std::string>();
            if (L.type == "conv1d") {
                L.activation = l.value("activation", "linear");
                L.padding = l.value("padding", "same");
                L.kernel_size = l.at("kernel_size").get<int>();
                L.filters = l.at("filters").get<int>();
                L.kernel3 = l.at("kernel").get<std::vector<std::vector<std::vector<double>>>>();
                L.bias = l.at("bias").get<std::vector<double>>();
            } else if (L.type == "maxpool1d") {
                L.pool_size = l.at("pool_size").get<int>();
                L.strides = l.value("strides", L.pool_size);
            } else if (L.type == "dense") {
                L.activation = l.value("activation", "linear");
                L.units = l.at("units").get<int>();
                L.kernel2 = l.at("kernel").get<std::vector<std::vector<double>>>();
                L.bias = l.at("bias").get<std::vector<double>>();
            }
            layers.push_back(L);
        }
    }

    // x: (length, channels), row-major. Returns class probabilities.
    std::vector<double> predict(const std::vector<double>& input) const {
        std::vector<std::vector<double>> a(input.size(), std::vector<double>(1));
        for (size_t i = 0; i < input.size(); i++) a[i][0] = input[i];

        for (const auto& L : layers) {
            if (L.type == "conv1d") {
                int len = static_cast<int>(a.size());
                int in_ch = static_cast<int>(a[0].size());
                int pad = (L.padding == "same") ? (L.kernel_size - 1) / 2 : 0;
                int out_len = (L.padding == "same") ? len : len - L.kernel_size + 1;
                std::vector<std::vector<double>> b(out_len,
                                                   std::vector<double>(L.filters, 0.0));
                for (int o = 0; o < out_len; o++)
                    for (int f = 0; f < L.filters; f++) {
                        double acc = L.bias[f];
                        for (int k = 0; k < L.kernel_size; k++) {
                            int idx = o + k - pad;
                            if (idx < 0 || idx >= len) continue;   // zero padding
                            for (int c = 0; c < in_ch; c++)
                                acc += a[idx][c] * L.kernel3[k][c][f];
                        }
                        b[o][f] = (L.activation == "relu") ? std::max(0.0, acc) : acc;
                    }
                a.swap(b);
            } else if (L.type == "maxpool1d") {
                int len = static_cast<int>(a.size());
                int ch = static_cast<int>(a[0].size());
                int out_len = (len - L.pool_size) / L.strides + 1;
                if (out_len < 0) out_len = 0;
                std::vector<std::vector<double>> b(out_len, std::vector<double>(ch));
                for (int o = 0; o < out_len; o++)
                    for (int c = 0; c < ch; c++) {
                        double m = -1e300;
                        for (int k = 0; k < L.pool_size; k++)
                            m = std::max(m, a[o * L.strides + k][c]);
                        b[o][c] = m;
                    }
                a.swap(b);
            } else if (L.type == "dropout") {
                // inference no-op, by definition
            } else if (L.type == "flatten") {
                std::vector<double> flat;
                for (const auto& row : a)                    // channels-last order
                    for (double v : row) flat.push_back(v);
                a.assign(flat.size(), std::vector<double>(1));
                for (size_t i = 0; i < flat.size(); i++) a[i][0] = flat[i];
            } else if (L.type == "dense") {
                std::vector<double> flat(a.size());
                for (size_t i = 0; i < a.size(); i++) flat[i] = a[i][0];
                std::vector<double> out(L.units);
                for (int u = 0; u < L.units; u++) {
                    double acc = L.bias[u];
                    for (size_t i = 0; i < flat.size(); i++) acc += flat[i] * L.kernel2[i][u];
                    out[u] = acc;
                }
                if (L.activation == "softmax") {
                    double mx = *std::max_element(out.begin(), out.end()), sum = 0.0;
                    for (double& v : out) { v = std::exp(v - mx); sum += v; }
                    for (double& v : out) v /= sum;
                } else if (L.activation == "relu") {
                    for (double& v : out) v = std::max(0.0, v);
                }
                a.assign(out.size(), std::vector<double>(1));
                for (size_t i = 0; i < out.size(); i++) a[i][0] = out[i];
            } else {
                throw std::runtime_error("unsupported layer type: " + L.type);
            }
        }
        std::vector<double> probs(a.size());
        for (size_t i = 0; i < a.size(); i++) probs[i] = a[i][0];
        return probs;
    }
};

// ── anomaly: Mahalanobis distance from the healthy centre ─────────────
// score = sqrt((z - mean) @ precision @ (z - mean)), higher = more anomalous.
struct AnomalyDetector {
    std::vector<double> mean;
    std::vector<std::vector<double>> precision;
    double threshold = 0.0;          // for a POOLED vector, ring depth pool_windows_assumed
    double threshold_single = 0.0;   // for ONE unpooled capture
    int pool_windows_assumed = 1;
    std::string direction = "higher_is_anomalous";
    std::string primary_model;

    void load(const std::string& model_dir, const std::string& config_dir) {
        nlohmann::json m;
        std::ifstream(model_dir + "/mahalanobis.json") >> m;
        mean = m.at("mean").get<std::vector<double>>();
        precision = m.at("precision").get<std::vector<std::vector<double>>>();
        threshold = m.at("threshold").get<double>();
        direction = m.value("direction", direction);
        threshold_single = threshold;

        // threshold.json is authoritative when its primary model is this one;
        // otherwise the Mahalanobis threshold from the model file is used and
        // the caller is told which scorer actually ran.
        std::ifstream tf(config_dir + "/threshold.json");
        if (tf) {
            nlohmann::json t;
            tf >> t;
            primary_model = t.value("primary_model", "");
            if (primary_model == "mahalanobis") threshold = t.value("threshold", threshold);
            pool_windows_assumed = t.value("pool_windows_assumed", 1);
            threshold_single = t.value("threshold_single_capture", threshold);
        }
    }

    // The ring medians FEATURES before scoring, so the score distribution has a
    // different shape at every ring depth and one threshold cannot serve both.
    // Measured 2026-08-20: the depth-20 threshold applied to a single capture
    // flags 30 % of healthy ones. Only depths 1 and pool_windows_assumed are
    // calibrated; anything else is a guess and says so.
    double threshold_for(int pool_windows) const {
        if (pool_windows <= 1) return threshold_single;
        return threshold;
    }
    bool threshold_is_calibrated_for(int pool_windows) const {
        return pool_windows <= 1 || pool_windows == pool_windows_assumed;
    }

    double score(const std::vector<double>& z) const {
        size_t n = mean.size();
        std::vector<double> d(n);
        for (size_t i = 0; i < n; i++) d[i] = z[i] - mean[i];
        double acc = 0.0;
        for (size_t i = 0; i < n; i++) {
            double row = 0.0;
            for (size_t k = 0; k < n; k++) row += precision[i][k] * d[k];
            acc += d[i] * row;
        }
        return std::sqrt(std::max(0.0, acc));
    }

    bool is_anomalous(double s) const { return s > threshold; }
    bool is_anomalous(double s, int pool_windows) const {
        return s > threshold_for(pool_windows);
    }
};

// ── RUL: scaler -> ridge -> health band -> remaining life ─────────────
//
// Two model shapes are accepted, decided by which keys the JSON carries.
//
//   "scaler+ridge+bands"     ridge output is a continuous SEVERITY, rounded to
//                            the nearest band; health and RUL are read from the
//                            band table. This is what rul.ipynb writes now.
//   "scaler+ridge+isotonic"  the older continuous model. Still loaded, because
//                            an older deployment must not stop working on a
//                            header change — but note the isotonic stage cannot
//                            output a value outside its fitted range, which is
//                            exactly wrong for judging a motor worse than
//                            anything it was trained on.
//
// Bands exist because the campaign steps resistance as 0.1/n ohms: the middle
// levels land 0.005-0.008 ohm apart and do not separate (AUC 0.497 for the
// closest pair), so a finer answer than a band would be invented.
struct RulModel {
    std::vector<double> coef;
    double intercept = 0.0;
    std::vector<double> iso_x, iso_y;                 // continuous model only
    std::vector<double> sev_x, sev_y;                 // optional severity calibration
    std::vector<std::string> band_names;              // band model only
    std::vector<double> band_health;
    double life_index = 0.0;
    std::string time_unit = "week";
    std::string status = "unknown";

    bool banded() const { return !band_health.empty(); }

    void load(const std::string& model_dir, const std::string& config_dir) {
        nlohmann::json j;
        std::ifstream(model_dir + "/health_model.json") >> j;
        coef = j.at("ridge").at("coef").get<std::vector<double>>();
        intercept = j.at("ridge").at("intercept").get<double>();
        life_index = j.at("life_index").get<double>();
        time_unit = j.value("time_unit", time_unit);

        if (j.contains("bands")) {
            band_names = j.at("bands").at("names").get<std::vector<std::string>>();
            band_health = j.at("bands").at("health").get<std::vector<double>>();
            if (band_names.size() != band_health.size())
                throw std::runtime_error("rul: band names and health differ in length");
            if (band_health.empty())
                throw std::runtime_error("rul: band table is empty");
        }
        if (j.contains("severity_calibration")) {
            sev_x = j.at("severity_calibration").at("x").get<std::vector<double>>();
            sev_y = j.at("severity_calibration").at("y").get<std::vector<double>>();
            if (sev_x.size() != sev_y.size())
                throw std::runtime_error("severity_calibration x and y differ in length");
        }
        if (j.contains("isotonic") && !j.contains("bands")) {
            iso_x = j.at("isotonic").at("x").get<std::vector<double>>();
            iso_y = j.at("isotonic").at("y").get<std::vector<double>>();
        }
        // Neither present is legal: a bare ridge, clipped to [0,1].

        std::ifstream cf(config_dir + "/rul_config.json");
        if (cf) {
            nlohmann::json c;
            cf >> c;
            status = c.value("status", status);
        }
    }

    // The ridge output before it is snapped to a band. Exposed because it says
    // how close a verdict sat to a boundary, which the band alone hides.
    // Monotone piecewise-linear lookup, linear outside the knots. Shared by the
    // severity calibration below and the isotonic health curve, which were two
    // copies of the same seven lines.
    static double interp(const std::vector<double>& xs, const std::vector<double>& ys,
                         double x, bool clamp_ends) {
        if (xs.size() < 2) return x;
        if (x <= xs.front())
            return clamp_ends ? ys.front()
                              : ys.front() + (x - xs.front()) *
                                    (ys[1] - ys[0]) / (xs[1] - xs[0]);
        if (x >= xs.back()) {
            size_t n = xs.size();
            return clamp_ends ? ys.back()
                              : ys.back() + (x - xs.back()) *
                                    (ys[n - 1] - ys[n - 2]) / (xs[n - 1] - xs[n - 2]);
        }
        size_t hi = std::lower_bound(xs.begin(), xs.end(), x) - xs.begin();
        size_t lo = hi ? hi - 1 : 0;
        double span = xs[hi] - xs[lo];
        double t = span > 0 ? (x - xs[lo]) / span : 0.0;
        return ys[lo] + t * (ys[hi] - ys[lo]);
    }

    // band() is round(severity), so the cuts are fixed at the half-integers and
    // the only way to give a band room is to move the severities. A linear
    // severity cannot do it: with four bands and only a scale and an offset,
    // widening one band's margin necessarily narrows its neighbour's. Measured
    // 2026-08-20 -- rebalancing took healthy from 0.031 to 0.194 of margin and
    // pushed 8Rp and 2Rp down in exchange, for no net gain on the device.
    //
    // sev_x/sev_y is an optional monotone curve that maps raw severity onto the
    // band scale, with one knot per band, so every band centre can sit on its
    // own integer at once. Absent from the JSON, this is a no-op and the model
    // behaves exactly as before.
    double severity(const std::vector<double>& z) const {
        double raw = intercept;
        for (size_t i = 0; i < coef.size() && i < z.size(); i++) raw += coef[i] * z[i];
        if (sev_x.size() >= 2) return interp(sev_x, sev_y, raw, /*clamp_ends=*/false);
        return raw;
    }

    int band(const std::vector<double>& z) const {
        if (!banded()) return -1;
        double r = std::round(severity(z));
        if (r < 0) r = 0;
        if (r > static_cast<double>(band_health.size()) - 1)
            r = static_cast<double>(band_health.size()) - 1;
        return static_cast<int>(r);
    }

    std::string band_name(int b) const {
        return (b >= 0 && b < static_cast<int>(band_names.size())) ? band_names[b]
                                                                   : "unknown";
    }

    double health(const std::vector<double>& z) const {
        if (banded()) return band_health[static_cast<size_t>(band(z))];

        double raw = severity(z);
        if (iso_x.empty()) return std::min(1.0, std::max(0.0, raw));
        double h = interp(iso_x, iso_y, raw, /*clamp_ends=*/true);   // clipped, as in Python
        return std::min(1.0, std::max(0.0, h));
    }

    // Exact for the band model too: rul.ipynb defines a band's RUL as its
    // health times the life index, so there is one formula, not two.
    double remaining(double h) const { return std::max(0.0, h * life_index); }
};
