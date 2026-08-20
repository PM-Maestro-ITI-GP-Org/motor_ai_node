// feature_extraction.hpp
// C++ port of host_pipeline/data_building.ipynb Block 4.
//
// This file is the risky half of the system: the same features are computed
// here and in Python, and a model trained on one and served by the other is
// only as good as the agreement between them. Every formula below mirrors a
// named Python function, and test/test_parity.cpp checks the numbers against
// values exported from the notebook.
//
// Amplitude features use the window; frequency-hungry ones (RPM, MCSA, spectral
// ratios, order domain) use the longer context slice, exactly as in Python.
//
// kissfft is built with kiss_fft_scalar=double (see CMakeLists) so its results
// match numpy's float64 rather than drifting at the 1e-6 level.
#pragma once

#include <algorithm>
#include <cmath>
#include <complex>
#include <map>
#include <numeric>
#include <string>
#include <vector>

#include "extraction_config.hpp"
#include "kiss_fftr.h"

using FeatureMap = std::map<std::string, double>;
using Channels   = std::map<std::string, std::vector<double>>;

struct Spectrum {
    std::vector<double> freqs;   // Hz
    std::vector<double> mag;     // |rfft|, mean removed first
};

// Magnitude spectrum of a mean-removed signal — the C++ side of compute_fft().
// Even lengths go through kissfft; odd lengths take the direct path below,
// because scipy.fft.rfft accepts any length and kiss_fftr does not.
inline Spectrum compute_fft(const std::vector<double>& sig, double fs) {
    Spectrum out;
    int n = static_cast<int>(sig.size());
    if (n < 4) return out;

    if (n % 2 != 0) {
        // Direct rfft. kiss_fftr requires an even length; numpy does not, and the
        // order-domain resampler produces odd lengths routinely. Truncating to
        // even there changed the spectrum enough to move order_2x by 70%. These
        // arrays are ~100 samples, so O(n^2) costs nothing.
        double mean_odd = std::accumulate(sig.begin(), sig.end(), 0.0) / n;
        int bins = n / 2 + 1;
        out.freqs.resize(bins);
        out.mag.resize(bins);
        for (int k = 0; k < bins; k++) {
            double re = 0.0, im = 0.0;
            for (int t = 0; t < n; t++) {
                double angle = -2.0 * M_PI * k * t / n;
                double v = sig[t] - mean_odd;
                re += v * std::cos(angle);
                im += v * std::sin(angle);
            }
            out.freqs[k] = static_cast<double>(k) * fs / n;
            out.mag[k] = std::hypot(re, im);
        }
        return out;
    }

    double mean = std::accumulate(sig.begin(), sig.begin() + n, 0.0) / n;
    std::vector<kiss_fft_scalar> in(n);
    for (int i = 0; i < n; i++) in[i] = static_cast<kiss_fft_scalar>(sig[i] - mean);

    std::vector<kiss_fft_cpx> spec(n / 2 + 1);
    kiss_fftr_cfg cfg = kiss_fftr_alloc(n, 0, nullptr, nullptr);
    kiss_fftr(cfg, in.data(), spec.data());
    kiss_fftr_free(cfg);

    out.freqs.resize(n / 2 + 1);
    out.mag.resize(n / 2 + 1);
    for (int i = 0; i <= n / 2; i++) {
        out.freqs[i] = static_cast<double>(i) * fs / n;
        out.mag[i] = std::hypot(spec[i].r, spec[i].i);
    }
    return out;
}

inline double band_energy(const Spectrum& s, double lo, double hi) {
    double acc = 0.0;
    for (size_t i = 0; i < s.freqs.size(); i++)
        if (s.freqs[i] >= lo && s.freqs[i] <= hi) acc += s.mag[i] * s.mag[i];
    return acc;
}

inline double peak_near(const Spectrum& s, double target, double hw) {
    if (target <= 0) return 0.0;
    double best = 0.0;
    bool found = false;
    for (size_t i = 0; i < s.freqs.size(); i++)
        if (s.freqs[i] >= target - hw && s.freqs[i] <= target + hw) {
            best = std::max(best, s.mag[i]);
            found = true;
        }
    return found ? best : 0.0;
}

struct RpmEstimate { double rpm = 0.0, fe = 0.0; };

// Band-limited on purpose: the tallest peak in this motor's current spectrum is
// the PWM switching frequency, which reports tens of thousands of rpm.
inline RpmEstimate estimate_rpm(const std::vector<double>& I1, const ExtractionConfig& c) {
    RpmEstimate r;
    if (I1.size() < 8) return r;
    double mean = std::accumulate(I1.begin(), I1.end(), 0.0) / I1.size();
    double var = 0.0;
    for (double v : I1) var += (v - mean) * (v - mean);
    if (std::sqrt(var / I1.size()) < 1e-6) return r;

    Spectrum s = compute_fft(I1, c.fs);
    if (s.mag.size() < 3) return r;

    int k = -1;
    double best = -1.0;
    for (size_t i = 1; i < s.freqs.size(); i++)
        if (s.freqs[i] >= c.rpm_lo && s.freqs[i] <= c.rpm_hi && s.mag[i] > best) {
            best = s.mag[i];
            k = static_cast<int>(i);
        }
    if (k < 0) {                                     // band narrower than one bin
        for (size_t i = 1; i < s.mag.size(); i++)
            if (s.mag[i] > best) { best = s.mag[i]; k = static_cast<int>(i); }
    }
    if (k <= 0) return r;

    double delta = 0.0;
    if (k >= 1 && k + 1 < static_cast<int>(s.mag.size())) {
        double a = s.mag[k - 1], b = s.mag[k], cc = s.mag[k + 1];
        delta = 0.5 * (a - cc) / (a - 2 * b + cc + c.eps);
    }
    r.fe = s.freqs[k] + delta * (s.freqs[1] - s.freqs[0]);
    r.rpm = r.fe / c.pole_pairs * 60.0;
    return r;
}

// ── time domain (19) ──────────────────────────────────────────────────
inline void extract_time_domain(const std::vector<double>& s, const std::string& prefix,
                                const ExtractionConfig& c, FeatureMap& out) {
    if (s.empty()) return;
    const double eps = c.eps;
    const size_t n = s.size();

    double sum = 0, sum_abs = 0, sum_sq = 0, sum_sqrt_abs = 0;
    double vmax = s[0], vmin = s[0], peak = 0;
    for (double v : s) {
        sum += v; sum_abs += std::fabs(v); sum_sq += v * v;
        sum_sqrt_abs += std::sqrt(std::fabs(v));
        vmax = std::max(vmax, v); vmin = std::min(vmin, v);
        peak = std::max(peak, std::fabs(v));
    }
    double mean = sum / n;
    double rms  = std::sqrt(sum_sq / n);
    double mabs = sum_abs / n + eps;

    double m2 = 0, m3 = 0, m4 = 0;
    for (double v : s) {
        double d = v - mean;
        m2 += d * d; m3 += d * d * d; m4 += d * d * d * d;
    }
    m2 /= n; m3 /= n; m4 /= n;
    double sd = std::sqrt(m2);

    std::vector<double> sorted(s.begin(), s.end());
    std::sort(sorted.begin(), sorted.end());
    auto percentile = [&](double p) {                 // numpy linear interpolation
        double idx = p / 100.0 * (sorted.size() - 1);
        size_t lo = static_cast<size_t>(std::floor(idx));
        size_t hi = static_cast<size_t>(std::ceil(idx));
        return sorted[lo] + (sorted[hi] - sorted[lo]) * (idx - lo);
    };
    double median = percentile(50.0);
    std::vector<double> dev(n);
    for (size_t i = 0; i < n; i++) dev[i] = std::fabs(s[i] - median);
    std::sort(dev.begin(), dev.end());
    double mad = dev[n / 2];
    if (n % 2 == 0) mad = 0.5 * (dev[n / 2 - 1] + dev[n / 2]);

    size_t crossings = 0;
    for (size_t i = 1; i < n; i++)
        if (std::signbit(s[i] - mean) != std::signbit(s[i - 1] - mean)) crossings++;

    const auto on = [&](const char* k) {
        auto it = c.time_features.find(k);
        return it == c.time_features.end() || it->second;
    };
    if (on("mean"))      out[prefix + "mean"] = mean;
    if (on("abs_mean"))  out[prefix + "abs_mean"] = sum_abs / n;
    if (on("rms"))       out[prefix + "rms"] = rms;
    if (on("std"))       out[prefix + "std"] = sd;
    if (on("variance"))  out[prefix + "variance"] = m2;
    if (on("max"))       out[prefix + "max"] = vmax;
    if (on("min"))       out[prefix + "min"] = vmin;
    if (on("peak"))      out[prefix + "peak"] = peak;
    if (on("p2p"))       out[prefix + "p2p"] = vmax - vmin;
    if (on("crest"))     out[prefix + "crest"] = peak / (rms + eps);
    // Pearson kurtosis (scipy fisher=False) — a Fisher one would be off by 3.
    // No epsilon in these denominators: scipy has none, and for a channel with
    // variance ~1e-4 an eps of 1e-9 is 10% of m2^2. Guard against the constant
    // signal instead, which is the only case the epsilon was protecting.
    if (on("kurtosis"))  out[prefix + "kurtosis"] = m2 > 0 ? m4 / (m2 * m2) : 0.0;
    if (on("skewness"))  out[prefix + "skewness"] = m2 > 0 ? m3 / std::pow(m2, 1.5) : 0.0;
    if (on("shape"))     out[prefix + "shape"] = rms / mabs;
    if (on("impulse"))   out[prefix + "impulse"] = peak / mabs;
    if (on("clearance"))
        out[prefix + "clearance"] = peak / (std::pow(sum_sqrt_abs / n, 2) + eps);
    if (on("energy"))    out[prefix + "energy"] = sum_sq;
    if (on("zcr"))       out[prefix + "zcr"] = n > 1 ? static_cast<double>(crossings) / (n - 1) : 0.0;
    if (on("mad"))       out[prefix + "mad"] = mad;
    if (on("iqr"))       out[prefix + "iqr"] = percentile(75.0) - percentile(25.0);
}

// ── frequency domain (11 + one per band) ──────────────────────────────
inline void extract_freq_domain(const std::vector<double>& sig, double fs,
                                const std::string& prefix, const ExtractionConfig& c,
                                FeatureMap& out) {
    if (sig.size() < 8) return;
    Spectrum s = compute_fft(sig, fs);
    if (s.mag.size() < 3) return;
    const double eps = c.eps;

    std::vector<double> power(s.mag.size());
    double total = 0.0, mag_sum = 0.0, mag_max = 0.0;
    for (size_t i = 0; i < s.mag.size(); i++) {
        power[i] = s.mag[i] * s.mag[i];
        total += power[i];
        mag_sum += s.mag[i];
        mag_max = std::max(mag_max, s.mag[i]);
    }
    total += eps;

    size_t di = 1;
    for (size_t i = 1; i < s.mag.size(); i++) if (s.mag[i] > s.mag[di]) di = i;

    const auto on = [&](const char* k) {
        auto it = c.freq_features.find(k);
        return it == c.freq_features.end() || it->second;
    };
    if (on("dominant_freq")) out[prefix + "dominant_freq"] = s.freqs[di];
    if (on("dominant_mag"))  out[prefix + "dominant_mag"] = s.mag[di];
    if (on("spectral_energy")) out[prefix + "spectral_energy"] = total - eps;
    if (on("spectral_entropy")) {
        // Normalised by bin count, not non-zero bin count — see the note in the
        // notebook. Zero bins contribute nothing (p*log p -> 0) and are skipped;
        // no epsilon inside the log, which would otherwise become the value for
        // very small bins.
        double h = 0.0;
        for (double p : power) {
            double pi = p / total;
            if (pi > 0) h -= pi * std::log2(pi);
        }
        out[prefix + "spectral_entropy"] = h / std::log2(static_cast<double>(power.size()));
    }
    if (on("spectral_flatness")) {
        double log_sum = 0.0;
        for (double p : power) log_sum += std::log(p + eps);
        double gm = std::exp(log_sum / power.size());
        double am = (total - eps) / power.size();
        out[prefix + "spectral_flatness"] = gm / (am + eps);
    }
    if (on("thd")) {
        double dom = s.freqs[di];
        double a1 = peak_near(s, dom, 2.0), h = 0.0;
        for (int k = 2; k <= 5; k++) {
            double a = peak_near(s, k * dom, 2.0);
            h += a * a;
        }
        out[prefix + "thd"] = std::sqrt(h) / (a1 + eps);
    }
    if (on("peak_to_mean"))
        out[prefix + "peak_to_mean"] = mag_max / (mag_sum / s.mag.size() + eps);
    if (on("rolloff")) {
        double cum = 0.0, target = 0.85 * (total - eps);
        size_t idx = 0;
        for (; idx < power.size(); idx++) { cum += power[idx]; if (cum >= target) break; }
        out[prefix + "rolloff"] = s.freqs[std::min(idx, s.freqs.size() - 1)];
    }
    double centroid = 0.0;
    for (size_t i = 0; i < power.size(); i++) centroid += s.freqs[i] * power[i];
    centroid /= total;
    if (on("centroid")) out[prefix + "centroid"] = centroid;
    if (on("spread")) {
        double acc = 0.0;
        for (size_t i = 0; i < power.size(); i++)
            acc += (s.freqs[i] - centroid) * (s.freqs[i] - centroid) * power[i];
        out[prefix + "spread"] = std::sqrt(acc / total);
    }
    if (on("slope")) {
        double fm = std::accumulate(s.freqs.begin(), s.freqs.end(), 0.0) / s.freqs.size();
        double mm = mag_sum / s.mag.size(), num = 0.0, den = 0.0;
        for (size_t i = 0; i < s.mag.size(); i++) {
            num += (s.freqs[i] - fm) * (s.mag[i] - mm);
            den += (s.freqs[i] - fm) * (s.freqs[i] - fm);
        }
        out[prefix + "slope"] = num / (den + eps);
    }
    if (on("bands")) {
        double nyq = fs / 2;
        for (auto [lo, hi] : c.bands) {
            if (lo >= nyq) continue;
            double hi_c = std::min(hi, nyq);
            out[prefix + "band_" + std::to_string(static_cast<long long>(lo)) + "_" +
                std::to_string(static_cast<long long>(hi_c))] = band_energy(s, lo, hi_c);
        }
    }
}

// ── MCSA (10) ─────────────────────────────────────────────────────────
inline void extract_mcsa(const std::vector<double>& I1, double rpm,
                         const ExtractionConfig& c, FeatureMap& out) {
    if (I1.size() < 8 || rpm <= 0) return;
    Spectrum s = compute_fft(I1, c.fs);
    if (s.freqs.size() < 3) return;

    double fr = rpm / 60.0, fe = c.pole_pairs * fr;
    // Half-width must be narrower than the sideband spacing or the search
    // around fe +/- fr also catches fe. fr/2 is the widest that cannot; the bin
    // width is the floor because no search resolves below it, and 2.0 Hz stays
    // the cap. Must match extract_mcsa in host_pipeline/data_building.ipynb --
    // this formula lives in two languages and there is no parity test running
    // to catch a drift (make_mock_data.py still emits the old logger format).
    double binw = s.freqs[1] - s.freqs[0];
    double hw = std::max(binw, std::min(2.0, fr / 2.0));
    double a_fe = peak_near(s, fe, hw);
    double a_lsb = peak_near(s, fe - fr, hw);
    double a_usb = peak_near(s, fe + fr, hw);

    out["mcsa_fundamental"] = a_fe;
    out["mcsa_lsb"] = a_lsb;
    out["mcsa_usb"] = a_usb;
    out["mcsa_sideband_ratio_db"] =
        20.0 * std::log10((a_lsb + a_usb + c.eps) / (a_fe + c.eps));
    out["mcsa_sideband_asymmetry"] =
        std::fabs(a_usb - a_lsb) / (a_lsb + a_usb + c.eps);
    double h = 0.0;
    for (int k = 2; k <= 5; k++) {
        double a_k = peak_near(s, k * fe, hw);
        out["mcsa_h" + std::to_string(k) + "_ratio"] = a_k / (a_fe + c.eps);
        h += a_k * a_k;
    }
    out["mcsa_thd"] = std::sqrt(h) / (a_fe + c.eps);
}

// ── phase balance (4 per prefix) ──────────────────────────────────────
inline void extract_phase_balance(const std::vector<double>& a, const std::vector<double>& b,
                                  const std::vector<double>& cc, const std::string& prefix,
                                  const ExtractionConfig& cfg, FeatureMap& out) {
    if (a.empty() || b.empty() || cc.empty()) return;
    auto rms = [](const std::vector<double>& v) {
        double acc = 0.0;
        for (double x : v) acc += x * x;
        return std::sqrt(acc / v.size());
    };
    double r[3] = {rms(a), rms(b), rms(cc)};
    double avg = (r[0] + r[1] + r[2]) / 3.0 + cfg.eps;
    for (int i = 0; i < 3; i++)
        out[prefix + "balance_" + std::to_string(i + 1)] = r[i] / avg;
    out[prefix + "imbalance"] =
        (*std::max_element(r, r + 3) - *std::min_element(r, r + 3)) / avg;
}

// ── spectral ratios (2) ───────────────────────────────────────────────
inline void extract_spectral_ratios(const std::vector<double>& I1, double rpm,
                                    const ExtractionConfig& c, FeatureMap& out) {
    if (I1.size() < 8 || rpm <= 0) return;
    Spectrum s = compute_fft(I1, c.fs);
    if (s.freqs.size() < 3) return;
    double fr = rpm / 60.0, fe = c.pole_pairs * fr;
    double binw = s.freqs[1] - s.freqs[0];      // same rule as extract_mcsa
    double hw = std::max(binw, std::min(2.0, fr / 2.0));
    double e_fe = band_energy(s, fe - hw, fe + hw) + c.eps;
    out["ratio_1xRPM_to_fe"] = band_energy(s, fr - hw, fr + hw) / e_fe;
    out["ratio_2x_to_fe"] = band_energy(s, 2 * fe - hw, 2 * fe + hw) / e_fe;
}

// ── order domain (3) ──────────────────────────────────────────────────
inline void extract_order_domain(const std::vector<double>& sig, double rpm, double fs,
                                 const ExtractionConfig& c, FeatureMap& out) {
    if (sig.size() < 8 || rpm <= 0) return;
    const int spr = 64;
    double fr = rpm / 60.0;
    int n_out = static_cast<int>(sig.size() / fs * fr * spr);
    if (n_out < 8) return;

    // np.interp over linspace(0, n-1, n_out) — the same resampling grid Python uses
    std::vector<double> r(n_out);
    double n_in = static_cast<double>(sig.size());
    for (int i = 0; i < n_out; i++) {
        double pos = (n_out == 1) ? 0.0 : (n_in - 1) * i / (n_out - 1.0);
        size_t lo = static_cast<size_t>(std::floor(pos));
        size_t hi = std::min(lo + 1, sig.size() - 1);
        r[i] = sig[lo] + (sig[hi] - sig[lo]) * (pos - lo);
    }
    Spectrum s = compute_fft(r, spr);
    if (s.freqs.size() < 4) return;

    double total = c.eps;
    for (double m : s.mag) total += m * m;
    for (int k = 1; k <= 3; k++)
        out["order_" + std::to_string(k) + "x"] =
            band_energy(s, k - 0.2, k + 0.2) / total;
}

// ── power (16) and DC bus (2) ─────────────────────────────────────────
inline void extract_power(const Channels& w, const ExtractionConfig& c, FeatureMap& out) {
    const char* iks[3] = {"I1", "I2", "I3"};
    const char* vks[3] = {"V1", "V2", "V3"};
    double p_total = 0.0;
    std::vector<double> zs, pfs;

    for (int n = 0; n < 3; n++) {
        auto ii = w.find(iks[n]), vi = w.find(vks[n]);
        if (ii == w.end() || vi == w.end()) continue;
        const auto& i_sig = ii->second;
        const auto& v_sig = vi->second;
        size_t m = std::min(i_sig.size(), v_sig.size());
        if (!m) continue;

        double i_sq = 0, v_sq = 0, p = 0;
        for (size_t k = 0; k < m; k++) {
            i_sq += i_sig[k] * i_sig[k];
            v_sq += v_sig[k] * v_sig[k];
            p += i_sig[k] * v_sig[k];
        }
        double i_rms = std::sqrt(i_sq / m), v_rms = std::sqrt(v_sq / m);
        double p_act = p / m, s_app = i_rms * v_rms;
        double pf = p_act / (s_app + c.eps), z = v_rms / (i_rms + c.eps);
        std::string tag = std::to_string(n + 1);
        out["P" + tag + "_active"] = p_act;
        out["S" + tag + "_apparent"] = s_app;
        out["PF" + tag] = pf;
        out["Z" + tag] = z;
        p_total += p_act;
        zs.push_back(z);
        pfs.push_back(pf);
    }
    if (zs.empty()) return;
    double z_mean = std::accumulate(zs.begin(), zs.end(), 0.0) / zs.size();
    out["P_total"] = p_total;
    out["Z_mean"] = z_mean;
    out["Z_imbalance"] =
        (*std::max_element(zs.begin(), zs.end()) - *std::min_element(zs.begin(), zs.end()))
        / (z_mean + c.eps);
    out["PF_mean"] = std::accumulate(pfs.begin(), pfs.end(), 0.0) / pfs.size();
}

inline void extract_dcbus(const std::vector<double>& vdc, const ExtractionConfig& c,
                          FeatureMap& out) {
    if (vdc.empty()) return;
    double mean = std::accumulate(vdc.begin(), vdc.end(), 0.0) / vdc.size();
    double vmin = *std::min_element(vdc.begin(), vdc.end());
    double var = 0.0;
    for (double v : vdc) var += (v - mean) * (v - mean);
    out["vdc_sag"] = (mean - vmin) / (std::fabs(mean) + c.eps);
    out["vdc_ripple"] = std::sqrt(var / vdc.size()) / (std::fabs(mean) + c.eps);
}

inline double channel_fs(const std::string& ch, const ExtractionConfig& c) {
    bool is_vib = std::find(c.vib_roles.begin(), c.vib_roles.end(), ch) != c.vib_roles.end();
    return (is_vib && c.vib_mode == "held") ? c.fs_vib : c.fs;
}

// Everything, for one window plus its context slice. Mirrors extract_window().
inline FeatureMap extract_all(const Channels& window, const Channels& context,
                              const ExtractionConfig& c, double& rpm_out, double& fe_out) {
    FeatureMap f;
    auto get = [](const Channels& m, const std::string& k) -> const std::vector<double>& {
        static const std::vector<double> empty;
        auto it = m.find(k);
        return it == m.end() ? empty : it->second;
    };
    const Channels& ctx = context.empty() ? window : context;

    RpmEstimate r = estimate_rpm(c.ctx_rpm ? get(ctx, "I1") : get(window, "I1"), c);
    rpm_out = r.rpm;
    fe_out = r.fe;

    if (c.groups.count("time") && c.groups.at("time"))
        for (const auto& ch : c.time_channels)
            if (window.count(ch)) extract_time_domain(get(window, ch), ch + "_", c, f);

    if (c.groups.count("freq") && c.groups.at("freq"))
        for (const auto& ch : c.freq_channels)
            if (window.count(ch))
                extract_freq_domain(get(window, ch), channel_fs(ch, c), ch + "_", c, f);

    const Channels& mcsa_src   = c.ctx_mcsa ? ctx : window;
    const Channels& ratios_src = c.ctx_ratios ? ctx : window;
    const Channels& order_src  = c.ctx_order ? ctx : window;

    if (c.groups.count("mcsa") && c.groups.at("mcsa"))
        extract_mcsa(get(mcsa_src, c.mcsa_channel), r.rpm, c, f);

    if (c.groups.count("phase") && c.groups.at("phase")) {
        extract_phase_balance(get(window, "I1"), get(window, "I2"), get(window, "I3"),
                              "i_", c, f);
        extract_phase_balance(get(window, "V1"), get(window, "V2"), get(window, "V3"),
                              "v_", c, f);
    }
    if (c.groups.count("ratios") && c.groups.at("ratios"))
        extract_spectral_ratios(get(ratios_src, c.mcsa_channel), r.rpm, c, f);

    if (c.groups.count("order") && c.groups.at("order") &&
        std::find(c.time_channels.begin(), c.time_channels.end(), c.order_channel)
            != c.time_channels.end())
        extract_order_domain(get(order_src, c.order_channel), r.rpm,
                             channel_fs(c.order_channel, c), c, f);

    if (c.groups.count("power") && c.groups.at("power")) extract_power(window, c, f);
    if (c.groups.count("dcbus") && c.groups.at("dcbus"))
        extract_dcbus(get(window, "vdc"), c, f);

    return f;
}
