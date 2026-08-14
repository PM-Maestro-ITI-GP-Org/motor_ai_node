// motor_ai_node -- the signal-driven half of the AI pipeline.
//
// It sleeps. motor_ai_server writes a completed window to
// <data_dir>/input_data/data.csv, signals it, and waits for
// <data_dir>/results/result.ini to appear:
//
//     SIGUSR1  ->  anomaly detection      ->  stage=anomaly
//     SIGUSR2  ->  fault classification   ->  stage=fault_class
//     SIGTERM  ->  remaining useful life  ->  stage=rul
//
// The models are not here yet. Every stage below reads the window, counts what
// it got and writes a fixed verdict, which is enough to exercise the whole
// handshake -- signal, wake, read, write, rename -- with the real file layout
// and the real timing. Replacing runAnomaly() and its two siblings with actual
// inference is the only change the rest of the system should need.
//
// Two things about this program are load-bearing rather than stylistic, and a
// real implementation has to keep both:
//
//   1. The result is written to a temporary name and renamed into place.
//      The server polls for the file's existence and reads it the moment it
//      appears; a result written in place would be read half-formed. rename(2)
//      within a directory is atomic, which is what makes "the file is there"
//      and "the file is complete" the same statement.
//
//   2. The pidfile is written only after the handlers are installed. The
//      server signals whatever pid it finds there, and the default
//      disposition of SIGTERM is to terminate -- publishing the pid first
//      means a window arriving in that gap kills the node.

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include <signal.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

namespace {

const char *kConfigEnv     = "MOTOR_AI_NODE_CONFIG";
const char *kConfigDefault = "/etc/motor-ai-node/node.conf";

// The three stages, in the order the server runs them. Used as an index into
// g_pending as well as the value written to stage= in result.ini.
enum Stage { kAnomaly = 0, kFaultClass = 1, kRul = 2, kStageCount = 3 };

const char *kStageName[kStageCount] = { "anomaly", "fault_class", "rul" };

struct NodeConfig {
    std::string dataDir{"/motor_data"};
    std::string pidFile{"/motor_data/ai.pid"};
    int         sig[kStageCount];

    NodeConfig()
    {
        sig[kAnomaly]    = SIGUSR1;
        sig[kFaultClass] = SIGUSR2;
        sig[kRul]        = SIGTERM;
    }
};

// Written by the handlers, read by the main loop. One flag per stage rather
// than one "which stage" variable: standard signals do not queue, so if two
// ever did arrive before the loop ran, a single variable would silently lose
// one of them.
volatile sig_atomic_t g_pending[kStageCount] = { 0, 0, 0 };
volatile sig_atomic_t g_running = 1;

// Handlers do nothing but record. Everything real -- reading the window,
// writing the result -- happens in the main loop, where it is allowed to call
// functions that are not async-signal-safe.
int g_sigToStage[NSIG];

void onStageSignal(int _sig)
{
    if (_sig > 0 && _sig < NSIG) {
        const int stage = g_sigToStage[_sig];
        if (stage >= 0) g_pending[stage] = 1;
    }
}

void onQuit(int) { g_running = 0; }

std::string trim(const std::string &_s)
{
    size_t b = _s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) return "";
    size_t e = _s.find_last_not_of(" \t\r\n");
    return _s.substr(b, e - b + 1);
}

// Same spelling as the server's parser, and the same set of names. Keeping the
// two in step matters more than it looks: a node listening on SIGUSR2 for a
// stage the server sends SIGRTMIN for waits forever, and neither side logs
// anything unusual until the timeout.
int parseSignal(const std::string &_s)
{
    if (_s.empty()) return -1;

    if (_s[0] >= '0' && _s[0] <= '9') {
        const long n = std::strtol(_s.c_str(), NULL, 10);
        return (n > 0 && n <= SIGRTMAX) ? static_cast<int>(n) : -1;
    }

    if (_s == "SIGUSR1") return SIGUSR1;
    if (_s == "SIGUSR2") return SIGUSR2;
    if (_s == "SIGTERM") return SIGTERM;
    if (_s == "SIGHUP")  return SIGHUP;
    if (_s == "SIGINT")  return SIGINT;

    if (_s.compare(0, 8, "SIGRTMIN") == 0) {
        int off = 0;
        if (_s.size() > 8) {
            if (_s[8] != '+') return -1;
            off = static_cast<int>(std::strtol(_s.c_str() + 9, NULL, 10));
        }
        const int sig = SIGRTMIN + off;
        return (sig >= SIGRTMIN && sig <= SIGRTMAX) ? sig : -1;
    }

    return -1;
}

bool mkdirp(const std::string &_path)
{
    if (_path.empty()) return false;

    std::string acc = (_path[0] == '/') ? "/" : "";
    size_t pos = (_path[0] == '/') ? 1 : 0;

    for (;;) {
        const size_t slash = _path.find('/', pos);
        const size_t end   = (slash == std::string::npos) ? _path.size() : slash;
        const std::string part = _path.substr(pos, end - pos);

        if (!part.empty()) {
            acc += part;
            if (mkdir(acc.c_str(), 0755) != 0 && errno != EEXIST) return false;
            acc += "/";
        }
        if (slash == std::string::npos) break;
        pos = slash + 1;
    }
    return true;
}

void loadConfig(NodeConfig &_cfg)
{
    const char *env = std::getenv(kConfigEnv);
    const std::string path = env ? env : kConfigDefault;

    std::ifstream f(path.c_str());
    if (!f.is_open()) {
        std::cerr << "[AI-node] no config at " << path << " -- using defaults" << std::endl;
        return;
    }

    std::string line;
    while (std::getline(f, line)) {
        const size_t hash = line.find('#');
        if (hash != std::string::npos) line = line.substr(0, hash);

        const size_t eq = line.find('=');
        if (eq == std::string::npos) continue;

        const std::string key = trim(line.substr(0, eq));
        const std::string val = trim(line.substr(eq + 1));
        if (key.empty() || val.empty()) continue;

        if      (key == "data_dir")    _cfg.dataDir = val;
        else if (key == "ai_pid_file") _cfg.pidFile = val;
        else if (key == "sig_anomaly" || key == "sig_fault_class" || key == "sig_rul") {
            const int sig = parseSignal(val);
            if (sig < 0) {
                std::cerr << "[AI-node] config: " << key << ": '" << val
                          << "' is not a signal -- keeping the default" << std::endl;
            } else if (key == "sig_anomaly")    _cfg.sig[kAnomaly]    = sig;
            else if  (key == "sig_fault_class") _cfg.sig[kFaultClass] = sig;
            else                                _cfg.sig[kRul]        = sig;
        }
        else std::cerr << "[AI-node] config: ignoring unknown key '" << key << "'" << std::endl;
    }
}

// ---------------------------------------------------------------------------
// The window
// ---------------------------------------------------------------------------
// Counting the rows is all a stub can honestly do with them, but it is not
// nothing: it proves the node is reading the window the server just wrote
// rather than a leftover, and a row count that does not match window_rows is
// the first symptom of the two sides disagreeing about the window size.
long countWindowRows(const std::string &_csvPath)
{
    std::ifstream f(_csvPath.c_str());
    if (!f.is_open()) {
        std::cerr << "[AI-node] could not read " << _csvPath << std::endl;
        return -1;
    }

    long lines = 0;
    std::string line;
    while (std::getline(f, line))
        if (!line.empty()) ++lines;

    return lines > 0 ? lines - 1 : 0;   // minus the header
}

// ---------------------------------------------------------------------------
// The models -- placeholders
// ---------------------------------------------------------------------------
// MOTOR_AI_FAKE_ANOMALY=1 makes this report an anomaly on every window, which
// is how the other two stages get exercised at all: the server only runs them
// when this one answers something other than "normal", so without a way to
// force it the fault-classification and RUL paths never execute on a healthy
// rig.
std::string runAnomaly(long _rows)
{
    std::cout << "[AI-node] anomaly detection over " << _rows << " rows" << std::endl;

    const char *fake = std::getenv("MOTOR_AI_FAKE_ANOMALY");
    return (fake && fake[0] == '1') ? "anomaly" : "normal";
}

std::string runFaultClass(long _rows)
{
    std::cout << "[AI-node] fault classification over " << _rows << " rows" << std::endl;
    return "unclassified";
}

std::string runRul(long _rows)
{
    std::cout << "[AI-node] remaining useful life over " << _rows << " rows" << std::endl;
    return "unknown";
}

// ---------------------------------------------------------------------------
// The result
// ---------------------------------------------------------------------------
// stage= is what lets the server tell this answer from a late one to the
// previous request; see readResult() on that side. It is not decoration.
bool writeResult(const std::string &_resultPath,
                 const char *_stage,
                 const std::string &_value,
                 long _rows)
{
    const std::string tmp = _resultPath + ".tmp";

    {
        std::ofstream f(tmp.c_str(), std::ios::trunc);
        if (!f.is_open()) {
            std::cerr << "[AI-node] could not open " << tmp << std::endl;
            return false;
        }

        f << "# written by motor_ai_node\n"
          << "stage = " << _stage << "\n"
          << "value = " << _value << "\n"
          << "rows = "  << _rows  << "\n";

        if (!f.good()) {
            std::cerr << "[AI-node] failed writing " << tmp << std::endl;
            return false;
        }
    }

    // The atomic step. Anything before this point is invisible to the server.
    if (std::rename(tmp.c_str(), _resultPath.c_str()) != 0) {
        std::cerr << "[AI-node] could not rename " << tmp << " to " << _resultPath
                  << ": " << std::strerror(errno) << std::endl;
        return false;
    }
    return true;
}

}  // namespace

int main()
{
    NodeConfig cfg;
    loadConfig(cfg);

    const std::string csvPath    = cfg.dataDir + "/input_data/data.csv";
    const std::string resultPath = cfg.dataDir + "/results/result.ini";

    if (!mkdirp(cfg.dataDir + "/input_data") || !mkdirp(cfg.dataDir + "/results")) {
        std::cerr << "[AI-node] ERROR: could not create " << cfg.dataDir
                  << ": " << std::strerror(errno) << std::endl;
        return 1;
    }

    // -- signal wiring ------------------------------------------------------
    for (int i = 0; i < NSIG; ++i) g_sigToStage[i] = -1;
    for (int s = 0; s < kStageCount; ++s) g_sigToStage[cfg.sig[s]] = s;

    // Blocked for the whole of main, and unblocked only inside sigsuspend().
    // This is what closes the window between testing g_pending and sleeping --
    // a signal arriving in that gap with a plain pause() would be recorded and
    // then slept through, and the server would time out waiting for a result
    // the node had already been asked for.
    sigset_t blocked, previous;
    sigemptyset(&blocked);
    for (int s = 0; s < kStageCount; ++s) sigaddset(&blocked, cfg.sig[s]);

    struct sigaction sa;
    std::memset(&sa, 0, sizeof sa);
    sa.sa_handler = onStageSignal;
    sa.sa_mask    = blocked;      // no stage handler interrupts another
    for (int s = 0; s < kStageCount; ++s) {
        if (sigaction(cfg.sig[s], &sa, NULL) != 0) {
            std::cerr << "[AI-node] could not install handler for signal "
                      << cfg.sig[s] << ": " << std::strerror(errno) << std::endl;
            return 1;
        }
    }

    // Shutdown. SIGTERM is deliberately absent: by default it is the RUL
    // request, so it cannot also mean "stop". That has a consequence outside
    // this file -- `systemctl stop` sends SIGTERM, so the unit has to set
    // KillSignal=SIGINT or stopping the service runs a model instead.
    std::memset(&sa, 0, sizeof sa);
    sa.sa_handler = onQuit;
    sigaction(SIGINT,  &sa, NULL);
    sigaction(SIGQUIT, &sa, NULL);

    sigprocmask(SIG_BLOCK, &blocked, &previous);

    // -- publish the pid, now that a signal cannot kill us ------------------
    {
        const std::string tmp = cfg.pidFile + ".tmp";
        std::ofstream f(tmp.c_str(), std::ios::trunc);
        if (!f.is_open()) {
            std::cerr << "[AI-node] ERROR: could not write " << tmp << std::endl;
            return 1;
        }
        f << getpid() << "\n";
        f.close();
        if (std::rename(tmp.c_str(), cfg.pidFile.c_str()) != 0) {
            std::cerr << "[AI-node] ERROR: could not rename " << tmp << " to "
                      << cfg.pidFile << ": " << std::strerror(errno) << std::endl;
            return 1;
        }
    }

    std::cout << "[AI-node] pid " << getpid() << " ready: "
              << "window=" << csvPath << " result=" << resultPath
              << " signals=" << cfg.sig[kAnomaly]
              << "/" << cfg.sig[kFaultClass]
              << "/" << cfg.sig[kRul] << std::endl;

    // -- the loop -----------------------------------------------------------
    while (g_running) {
        bool any = false;
        for (int s = 0; s < kStageCount; ++s) any = any || g_pending[s];

        if (!any) {
            // Atomically unblocks, waits, and re-blocks. Returns once a
            // handler has run, which is the only thing that sets g_pending.
            sigsuspend(&previous);
            continue;
        }

        // Signals stay blocked through the work below, so the next request is
        // held pending rather than arriving mid-inference. The server sends
        // them one at a time and waits for each result, so this costs nothing
        // and removes a class of reentrancy question.
        for (int s = 0; s < kStageCount && g_running; ++s) {
            if (!g_pending[s]) continue;
            g_pending[s] = 0;

            const long rows = countWindowRows(csvPath);
            if (rows < 0) continue;   // no window: let the server time out

            std::string value;
            switch (s) {
                case kAnomaly:    value = runAnomaly(rows);    break;
                case kFaultClass: value = runFaultClass(rows); break;
                default:          value = runRul(rows);        break;
            }

            writeResult(resultPath, kStageName[s], value, rows);
        }
    }

    std::cout << "[AI-node] stopping" << std::endl;
    unlink(cfg.pidFile.c_str());
    return 0;
}
