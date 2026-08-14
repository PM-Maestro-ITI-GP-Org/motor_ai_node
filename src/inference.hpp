// inference.hpp -- the three real models, behind the three stage functions.
//
// The node's mechanics do not change: a signal sets a flag, the main loop reads
// <data_dir>/input_data/data.csv, calls one of these, and renames a result into
// place. All this adds is what happens between reading the window and returning
// a string.
//
// Everything here is header-only and depends on nothing outside this tree:
// include/ carries the feature code and the JSON parser, kissfft/ the FFT,
// model/ and config/ the trained artefacts. That keeps the node's one promise
// -- one compiler, no configure step -- intact.
//
// Two things are worth knowing before changing anything below.
//
// 1. FEATURES ARE COMPUTED ONCE PER WINDOW, NOT ONCE PER STAGE. The server
//    signals anomaly, and then fault_class and rul for the SAME data.csv when
//    the first says something is wrong. Extracting three times would triple the
//    work and, worse, push one window into the ring three times. The cache is
//    keyed on the file's identity, so the second and third stages are free.
//
// 2. THE RING IS NOT AN OPTIMISATION. window_rows = 26000 is 1.3 s at 20 kHz,
//    which is one speed. The models were fitted on features pooled across a
//    full speed sweep, and current imbalance moves about ten times more with
//    speed than with the fault. Measured on six real windows of a FAILED motor,
//    scoring each window alone: anomaly detection 3/6, and one window reported
//    'early, 3.88 months' for a dead motor. Pooling the last pool_windows
//    windows restores the coverage the models were trained on.
//    Set pool_windows = 1 in node.conf for strict per-window scoring.
#pragma once

#include <algorithm>
#include <cmath>
#include <deque>
#include <iostream>
#include <string>
#include <sys/stat.h>
#include <vector>

#include "extraction_config.hpp"
#include "feature_extraction.hpp"
#include "io_utils.hpp"
#include "models.hpp"

namespace inference {

// np.median: the mean of the two middle values on an even count. The host pools
// with np.median, so anything else here is a silent disagreement with training.
inline double median(std::vector<double> v)
{
    if (v.empty()) return 0.0;
    const size_t mid = v.size() / 2;
    std::nth_element(v.begin(), v.begin() + mid, v.end());
    const double hi = v[mid];
    if (v.size() % 2) return hi;
    const double lo = *std::max_element(v.begin(), v.begin() + mid);
    return 0.5 * (lo + hi);
}

class Engine {
public:
    // Loaded once at startup. A missing or malformed artefact is fatal here
    // rather than at the first signal: a node that answers "unknown" forever
    // looks like the server's fault, and the server has no way to tell.
    bool load(const std::string &_root, int _poolWindows, int _windowsPerRead)
    {
        poolWindows_ = _poolWindows > 0 ? _poolWindows : 1;
        windowsPerRead_ = _windowsPerRead > 0 ? _windowsPerRead : 20;
        try {
            cfg_ = load_extraction_config(_root + "/config/feature_extraction.json");

            anomalyIn_.load(_root + "/config/anomaly");
            anomaly_.load(_root + "/model/anomaly", _root + "/config/anomaly");

            clfIn_.load(_root + "/config/classification");
            clf_.load(_root + "/model/classification");

            rulIn_.load(_root + "/config/rul");
            rul_.load(_root + "/model/rul", _root + "/config/rul");
        } catch (const std::exception &e) {
            std::cerr << "[AI-node] model load failed: " << e.what() << "\n"
                      << "[AI-node]   expected model/ and config/ under " << _root
                      << std::endl;
            return false;
        }

        std::cout << "[AI-node] models loaded from " << _root
                  << ": fs=" << static_cast<int>(cfg_.fs)
                  << " window=" << cfg_.window_samples
                  << " pool=" << windowsPerRead_ << " per read"
                  << " ring=" << poolWindows_ << " window(s)"
                  << " status=" << rul_.status << std::endl;
        return true;
    }

    // Extract this window's features if the file has changed since last time,
    // push them into the ring, and pool the ring. Returns false when the window
    // cannot be used, which the caller turns into "let the server time out"
    // rather than into a wrong answer.
    bool refresh(const std::string &_csvPath)
    {
        struct stat st;
        if (stat(_csvPath.c_str(), &st) != 0) {
            std::cerr << "[AI-node] cannot stat " << _csvPath << std::endl;
            return false;
        }
        // Same window as last time: the server is asking for a later stage of
        // the request we already extracted. Reuse it, and do NOT push it into
        // the ring a second time.
        if (st.st_ino == lastIno_ && st.st_mtime == lastMtime_ &&
            st.st_size == lastSize_ && !pooled_.empty())
            return true;

        FeatureMap f;
        if (!extractOne(_csvPath, f)) return false;

        ring_.push_back(f);
        while (static_cast<int>(ring_.size()) > poolWindows_) ring_.pop_front();
        pooled_ = poolRing();

        lastIno_ = st.st_ino;
        lastMtime_ = st.st_mtime;
        lastSize_ = st.st_size;
        return true;
    }

    std::string runAnomaly()
    {
        std::vector<std::string> missing;
        const double score = anomaly_.score(anomalyIn_.vectorise(pooled_, &missing));
        warnMissing("anomaly", missing);
        const bool bad = anomaly_.is_anomalous(score);
        std::cout << "[AI-node] anomaly " << (bad ? "anomaly" : "normal")
                  << "  score " << score << " vs " << anomaly_.threshold
                  << "  (" << ring_.size() << " window(s) pooled)" << std::endl;
        return bad ? "anomaly" : "normal";
    }

    std::string runFaultClass()
    {
        std::vector<std::string> missing;
        const auto probs = clf_.predict(clfIn_.vectorise(pooled_, &missing));
        warnMissing("classification", missing);
        size_t best = 0;
        for (size_t i = 1; i < probs.size(); ++i)
            if (probs[i] > probs[best]) best = i;
        std::cout << "[AI-node] fault_class " << clf_.classes[best]
                  << "  confidence " << probs[best] << std::endl;
        return clf_.classes[best];
    }

    // Returns the pred_maint string, which is what predMaintResult carries on
    // the wire. The band leads because the model cannot resolve severity finer
    // than a band -- the middle resistance levels do not separate in the data --
    // and "provisional" travels from the training config to the cluster so a
    // number built on one motor is never mistaken for a measurement.
    std::string runRul()
    {
        std::vector<std::string> missing;
        const auto z = rulIn_.vectorise(pooled_, &missing);
        warnMissing("rul", missing);

        const double health = rul_.health(z);
        const double remaining = rul_.remaining(health);
        const std::string band = rul_.banded() ? rul_.band_name(rul_.band(z))
                                               : std::string();

        std::cout << "[AI-node] rul " << (band.empty() ? "" : band + "  ")
                  << "health " << health << " -> " << remaining << " "
                  << rul_.time_unit << "s [" << rul_.status << "]" << std::endl;

        std::string out = band.empty() ? "" : band + ", ";
        out += "health " + trim(health) + ", RUL " + trim(remaining) + " " +
               rul_.time_unit + "s";
        if (rul_.status == "provisional") out += " (provisional)";
        return out;
    }

private:
    static std::string trim(double v)
    {
        std::string s = std::to_string(v);
        // std::to_string gives six decimals; two is what a cluster can read.
        const size_t dot = s.find('.');
        if (dot != std::string::npos && s.size() > dot + 3) s.erase(dot + 3);
        return s;
    }

    void warnMissing(const char *_task, const std::vector<std::string> &_missing) const
    {
        if (_missing.empty()) return;
        std::cerr << "[AI-node] " << _task << " model wants " << _missing.size()
                  << " feature(s) this build did not compute, e.g. "
                  << _missing.front() << std::endl;
    }

    // One data.csv -> one feature vector, pooled over the windows it contains.
    // Windows are spaced evenly across the file rather than taken from the
    // front, so a file that does span a range is represented end to end. That
    // is the same rule data_building uses when it limits windows per file.
    bool extractOne(const std::string &_csvPath, FeatureMap &_out)
    {
        auto table = load_csv_table(_csvPath);
        if (table.empty()) {
            std::cerr << "[AI-node] no rows in " << _csvPath << std::endl;
            return false;
        }
        Channels all = to_channels(table, cfg_, /*already_physical=*/false);
        if (!all.count("I1")) {
            std::cerr << "[AI-node] no current channel in " << _csvPath
                      << " -- check column_map in config/feature_extraction.json"
                      << std::endl;
            return false;
        }

        const size_t skip = startup_skip(all.at("I1").size(), cfg_, cfg_.skip_startup_ms);
        if (skip) all = slice(all, skip, all.at("I1").size() - skip);

        const size_t nRows = all.at("I1").size();
        if (static_cast<int>(nRows) < cfg_.window_samples) {
            std::cerr << "[AI-node] " << nRows << " rows after startup skip, need "
                      << cfg_.window_samples << std::endl;
            return false;
        }

        const int step = cfg_.step > 0 ? cfg_.step : cfg_.window_samples;
        std::vector<size_t> starts;
        for (size_t s = 0; s + static_cast<size_t>(cfg_.window_samples) <= nRows; s += step)
            starts.push_back(s);
        if (starts.empty()) return false;

        if (static_cast<int>(starts.size()) > windowsPerRead_) {
            std::vector<size_t> picks;
            const double span = static_cast<double>(starts.size() - 1);
            for (int i = 0; i < windowsPerRead_; ++i) {
                const double f = (windowsPerRead_ == 1)
                                     ? 0.0
                                     : span * i / (windowsPerRead_ - 1);
                picks.push_back(starts[static_cast<size_t>(std::llround(f))]);
            }
            starts = picks;
        }

        std::map<std::string, std::vector<double> > acc;
        int ok = 0;
        for (size_t s : starts) {
            Channels window = slice(all, s, static_cast<size_t>(cfg_.window_samples));
            Channels context;
            if (cfg_.context_samples > 0 &&
                nRows >= static_cast<size_t>(cfg_.context_samples)) {
                // Centred on the window and clamped, matching run_batch(). Off
                // centre context changes every spectral feature.
                const size_t half = static_cast<size_t>(cfg_.context_samples) / 2;
                const size_t mid = s + static_cast<size_t>(cfg_.window_samples) / 2;
                size_t c0 = (mid > half) ? mid - half : 0;
                c0 = std::min(c0, nRows - static_cast<size_t>(cfg_.context_samples));
                context = slice(all, c0, static_cast<size_t>(cfg_.context_samples));
            }
            double rpm = 0.0, fe = 0.0;
            FeatureMap f = extract_all(window, context, cfg_, rpm, fe);
            if (f.empty()) continue;
            for (std::map<std::string, double>::const_iterator it = f.begin();
                 it != f.end(); ++it)
                acc[it->first].push_back(it->second);
            ++ok;
        }
        if (ok == 0) {
            std::cerr << "[AI-node] every window rejected in " << _csvPath << std::endl;
            return false;
        }

        _out.clear();
        for (std::map<std::string, std::vector<double> >::iterator it = acc.begin();
             it != acc.end(); ++it)
            _out[it->first] = median(it->second);
        return true;
    }

    FeatureMap poolRing() const
    {
        if (ring_.size() == 1) return ring_.front();
        std::map<std::string, std::vector<double> > acc;
        for (std::deque<FeatureMap>::const_iterator w = ring_.begin();
             w != ring_.end(); ++w)
            for (FeatureMap::const_iterator it = w->begin(); it != w->end(); ++it)
                acc[it->first].push_back(it->second);
        FeatureMap out;
        for (std::map<std::string, std::vector<double> >::iterator it = acc.begin();
             it != acc.end(); ++it)
            out[it->first] = median(it->second);
        return out;
    }

    ExtractionConfig cfg_;
    TaskInput anomalyIn_, clfIn_, rulIn_;
    AnomalyDetector anomaly_;
    Classifier clf_;
    RulModel rul_;

    std::deque<FeatureMap> ring_;   // bounded by poolWindows_; ~2.5 kB each
    FeatureMap pooled_;
    int poolWindows_ = 20;
    int windowsPerRead_ = 20;

    ino_t lastIno_ = 0;
    time_t lastMtime_ = 0;
    off_t lastSize_ = -1;
};

}  // namespace inference
