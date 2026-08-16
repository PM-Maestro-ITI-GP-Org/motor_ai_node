// pipeline.hpp — raw channels to one pooled feature vector.
//
// This exists because there were three copies of "how a recording becomes a
// feature vector": main.cpp, test_parity.cpp, and the Python notebook. Two of
// them drifted. The parity test kept passing because it validated its own copy
// against Python's, while the binary that ships did something else — it took
// the context slice from a different place, and the disagreement only appears
// on windows that are not at position 0, which is the only window the test
// looked at.
//
// So: one function, called by main.cpp and by test_parity.cpp. A change here
// lands in the device and in the thing that checks the device at the same time.
// The Python half is host_pipeline/export_parity_case.py, which mirrors this
// deliberately and is the thing parity actually compares against.
#pragma once

#include <algorithm>
#include <cmath>
#include <map>
#include <string>
#include <vector>

#include "extraction_config.hpp"
#include "feature_extraction.hpp"
#include "io_utils.hpp"

// np.median: the mean of the two middle values on an even count. The host pools
// with np.median, so anything else here is a silent disagreement with training.
inline double median_of(std::vector<double> v) {
    if (v.empty()) return 0.0;
    const size_t mid = v.size() / 2;
    std::nth_element(v.begin(), v.begin() + mid, v.end());
    const double hi = v[mid];
    if (v.size() % 2) return hi;
    const double lo = *std::max_element(v.begin(), v.begin() + mid);
    return 0.5 * (lo + hi);
}

// Window starts, evenly spaced across everything available.
//
// Evenly spaced rather than the first N, so a varSpeed sweep is represented end
// to end instead of only at its starting speed. Same rule as
// LIMIT_WINDOWS_PER_FILE in data_building.
inline std::vector<size_t> window_starts(size_t n_rows, const ExtractionConfig& c,
                                         int max_windows) {
    const int step = c.step > 0 ? c.step : c.window_samples;
    std::vector<size_t> starts;
    for (size_t s = 0; s + static_cast<size_t>(c.window_samples) <= n_rows; s += step)
        starts.push_back(s);
    if (max_windows <= 0 || starts.size() <= static_cast<size_t>(max_windows))
        return starts;

    std::vector<size_t> picks;
    const double span = static_cast<double>(starts.size() - 1);
    for (int i = 0; i < max_windows; i++) {
        const double f = (max_windows == 1) ? 0.0 : span * i / (max_windows - 1);
        picks.push_back(starts[static_cast<size_t>(std::llround(f))]);
    }
    return picks;
}

// The context slice for one window: CONTEXT_SAMPLES centred on it, clamped to
// what is available.
//
// Centred and clamped, matching run_batch() in data_building. The older code
// here took the first CONTEXT_SAMPLES rows regardless of which window was being
// described, which agrees with the centred version only for the window at
// position 0 — and position 0 was the only window the parity test checked.
// Every spectral feature (rpm, MCSA, ratios, order) reads this slice, so
// getting it from the wrong place changes them all.
inline Channels context_for(const Channels& all, size_t start, size_t n_rows,
                            const ExtractionConfig& c) {
    if (c.context_samples <= 0 || n_rows < static_cast<size_t>(c.context_samples))
        return Channels();
    const size_t half = static_cast<size_t>(c.context_samples) / 2;
    const size_t mid = start + static_cast<size_t>(c.window_samples) / 2;
    size_t c0 = (mid > half) ? mid - half : 0;
    c0 = std::min(c0, n_rows - static_cast<size_t>(c.context_samples));
    return slice(all, c0, static_cast<size_t>(c.context_samples));
}

struct PooledResult {
    FeatureMap features;
    double rpm = 0.0, fe = 0.0;
    int n_pooled = 0;        // windows that produced features
    size_t n_available = 0;  // windows the input could have offered
};

// Extract features for every selected window and median-pool them.
//
// `all` must already have had the startup skip applied. Returns n_pooled == 0
// when nothing usable came out, which callers should treat as "no verdict"
// rather than as a verdict.
inline PooledResult pooled_features(const Channels& all, const ExtractionConfig& c,
                                    int max_windows) {
    PooledResult out;
    if (!all.count("I1")) return out;
    const size_t n_rows = all.at("I1").size();

    const std::vector<size_t> starts_all = window_starts(n_rows, c, 0);
    out.n_available = starts_all.size();
    const std::vector<size_t> starts = window_starts(n_rows, c, max_windows);

    std::map<std::string, std::vector<double> > acc;
    double rpm_sum = 0.0, fe_sum = 0.0;
    for (size_t s : starts) {
        Channels window = slice(all, s, static_cast<size_t>(c.window_samples));
        Channels context = context_for(all, s, n_rows, c);
        double rpm_i = 0.0, fe_i = 0.0;
        FeatureMap f = extract_all(window, context, c, rpm_i, fe_i);
        if (f.empty()) continue;
        for (FeatureMap::const_iterator it = f.begin(); it != f.end(); ++it)
            acc[it->first].push_back(it->second);
        rpm_sum += rpm_i;
        fe_sum += fe_i;
        out.n_pooled++;
    }
    if (out.n_pooled == 0) return out;

    for (std::map<std::string, std::vector<double> >::iterator it = acc.begin();
         it != acc.end(); ++it)
        out.features[it->first] = median_of(it->second);
    out.rpm = rpm_sum / out.n_pooled;
    out.fe = fe_sum / out.n_pooled;
    return out;
}
