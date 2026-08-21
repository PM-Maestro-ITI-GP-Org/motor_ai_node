# motor_ai_node

The inference half of the QNX motor demo's AI pipeline. It runs on the Linux
guest (guest-2), beside [motor_ai_server][server], and sleeps until that
service asks it for a verdict.

**Verified on hardware 2026-08-21, tagged `v0.1.0-rig`.** On an aarch64 target
against a live motor with 0.1 Ω of added series resistance — the most severe
fault in the training set — all three stages agreed: `anomaly` / `electrical` /
`failed`, with an anomaly score of 29–36 sitting inside the measured 1Rp
distribution (p5 21.2, median 27.6, p95 41.8). It places the fault at the right
severity, not merely detects it.

Three real models, no ML runtime: anomaly detection by Mahalanobis distance on
four current-balance features, binary fault classification (`normal` vs
`electrical`) as a one-layer network, and remaining useful life by a ridge onto
four health bands with a piecewise severity calibration. They are JSON, evaluated by plain arithmetic in
`include/models.hpp` — no TensorFlow, no scikit-learn, nothing to install.
Trained by [the AI repository][ai]; `model/` and `config/` here are the pinned
copies that ship.

[ai]: https://github.com/PM-Maestro-ITI-GP-Org/AI

[server]: https://github.com/PM-Maestro-ITI-GP-Org/motor_ai_server
[client]: https://github.com/PM-Maestro-ITI-GP-Org/motor_ai_client

## Where it sits

```
  QNX guest (guest-1)                  Linux guest (guest-2)
  ┌──────────────────┐                 ┌──────────────────┐   ┌───────────────┐
  │ motor_ai_client  │  sendBatch      │ motor_ai_server  │   │ motor_ai_node │
  │                  │ ──────────────► │                  │   │               │
  │  reads /motor_   │   26000 rows    │  writes          │   │   sleeps in   │
  │  ctrl shm        │   over TCP      │  data.csv        │   │   sigsuspend  │
  │                  │                 │        │         │   │       ▲       │
  │                  │                 │        └─signal──┼──►│───────┘       │
  │                  │                 │                  │   │       │       │
  │                  │ ◄────────────── │  polls result.ini│◄──┼───────┘       │
  │  publishes to    │  3 verdicts     │                  │   │  writes it    │
  │  /motor_ai_result│                 └──────────────────┘   └───────────────┘
  └──────────────────┘
```

The two processes meet in one directory, `data_dir`, which defaults to
`/motor_data`:

    /motor_data/input_data/data.csv    one completed window, written by the server
    /motor_data/results/request.ini    the id being asked for, written by the server
    /motor_data/results/result.ini     one stage's verdict, written by this
    /motor_data/ai.pid                 this process's pid, for the server to signal

## The protocol

| signal (default) | stage                 | `stage=` written | asked for when     |
|------------------|-----------------------|------------------|--------------------|
| `SIGUSR1`        | anomaly detection     | `anomaly`        | every window       |
| `SIGUSR2`        | fault classification  | `fault_class`    | anomaly != normal  |
| `SIGTERM`        | remaining useful life | `rul`            | anomaly != normal  |

The server writes `request.ini`, sends one signal, waits for
`results/result.ini` to appear, and checks that both `stage=` and `id=` are the
ones it asked for. A healthy motor costs one model; only a suspect one pays for
all three. RUL is also asked for on a timer, independently of any anomaly.

`request.ini`, written by the server before it signals:

```ini
# written by motor_ai_server
id = 42
stage = anomaly
```

`result.ini`, written by this — same `key = value` syntax as everything else
the two exchange:

```ini
# written by motor_ai_node
stage = anomaly
id = 42
value = normal
rows = 26000
```

`id=` is echoed, not interpreted. Which model runs is decided by which signal
arrived, never by this file. It exists because `stage=` alone cannot catch a
late answer to a timed-out request for the *same* stage — which has the right
stage name and a genuinely new file, and was computed from a window that is
already gone. If `request.ini` is absent the node echoes `id = 0`, which the
server rejects.

## Why the window is pooled

`window_rows = 26000` is **1.3 seconds** at 20 kHz, and at 1.3 seconds the motor
is at one speed. The models were fitted on features pooled across a full speed
sweep, and current imbalance moves about ten times more with speed than it does
with the fault — so a window from one speed reads as a different motor.

Measured on real recordings, one 26 000-row window at a time:

| | failed motor | healthy motor |
|---|---|---|
| `pool_windows = 1` | anomaly detected **3 of 6**; one window reported `early, 3.88 months` for a dead motor | 1 false alarm in 6 |
| `pool_windows = 20` | anomaly detected **19 of 20**; RUL `failed` from the second window on | 1 false alarm in 20 |

So the node keeps a ring of the last `pool_windows` feature vectors and scores
the median of the ring. Nothing about the protocol changes — the server, the
client and `window_rows` are untouched — and the ring holds features, not
samples, so it costs about 50 kB. Peak RSS is ~11 MB either way, dominated by
parsing the CSV.

The cost is that a verdict describes the motor over the last ~26 s rather than
strictly the window just received. For RUL that is the intended reading; the
server's own `rul_interval_ms` documentation makes the same argument. Set
`pool_windows = 1` for strict per-window scoring.

## The column map does not match the server's header, on purpose

`motor_ai_server` writes its CSV columns in **training order** but emits a header
naming them in a **different** order. This node maps channels by name, so
`config/feature_extraction.json` compensates:

| the header says | it actually carries |
|---|---|
| `Volt_0` | the **speed command** |
| `Volt_1` / `Volt_2` / `DC_bus_volt` | `Volt_0` / `Volt_1` / `Volt_2` |
| `Speed_volt_cmd` | the **DC bus** |

Confirmed on a live 26 000-row window: relabelled in training order all eight
electrical columns match the rig profile, and the column called
`Speed_volt_cmd` is steady at 2633 ± 13 — a DC bus, not a command.

**Currents occupy the same positions in both orders.** That is the whole
diagnostic story: the anomaly stage uses four balance features and nothing else,
so it was right throughout, while classification and RUL — which need `PF` and
`Z` from the voltages — were computing power factor from the speed command.
`normal, confidence 1.0` on a failed motor was a logistic regression saturating
on nonsense.

**IF THE SERVER IS EVER FIXED, THIS FILE MUST BE REVERTED** to the training
order or the node breaks identically and silently. The training-order map is
kept in the file as `column_map_training_order`, and the reasoning beside it in
`column_map_note`. The real fix is one line upstream, and the server also omits
`seq` — which is how dropped samples would be detected, and exactly the class of
bug it would have caught.

## Known hole: a stopped motor gets a confident answer

With the shaft still, every feature is a ratio of phase currents computed on
noise. `i_imbalance` reads **0.188** on a stopped 1Rp motor against **0.179** for
the same fault running, so the garbage lands inside the range of a real reading.
All three stages answer and disagree: `anomaly`, `normal`, `healthy` — on a
failed motor.

There is no guard in `v0.1.0-rig`. One was written and measured (refuse below a
minimum phase-current RMS; 7.96 % of blocks refused, median recording still
93.1 % usable) and dropped as unwanted.

## The anomaly stage is not trustworthy on one window yet

Measured on the current models, same healthy recording, same threshold (8.18):

| what the node is given | anomaly score | verdict |
|---|---|---|
| the whole 288 s recording | 5.06 | normal — correct |
| one 26 000-row window (1.3 s) | 108–127 | **anomaly — wrong** |

Raising `pool_windows` does not fix it: the score stays at 110–127 with a full
ring of 20, because every entry in the ring is itself a narrow capture.

The reason is the model, not the pooling. The anomaly detector is a Mahalanobis
distance fitted on **one** healthy recording — three healthy recordings split
60/20/20 by session leaves exactly one in development — and that distance grows
very fast off-distribution. A vector pooled across a whole speed sweep averages
the session-to-session difference away and lands near the fitted cloud; any
short capture at one speed does not, and the distance explodes.

So on the device today: **the fault classifier and the RUL band are reliable on
a 1.3 s window, and the anomaly stage is not.** RUL is unaffected because a
ridge fit is linear and degrades gracefully; it reads the correct band on every
recording tested.

What fixes it, in order of effect:

1. **More healthy recordings, from more sessions.** With four or more the split
   can fit on two sessions, so the model learns what day-to-day variation looks
   like instead of treating it as a fault.
2. **Fit the anomaly model on blocks that match what the device sees** — short,
   single-speed captures rather than sweep-pooled ones. Training and inference
   currently use different representations, and Mahalanobis is the least
   forgiving model in the set about that.

Until one of those lands, treat `anomalyResult` as advisory and read
`faultClassResult` and `predMaintResult` on their own. Note that
`motor_ai_server` only asks for those two when the anomaly stage says something
other than `normal`, so a false alarm costs two extra model runs rather than a
wrong answer — the failure direction is the safe one.

### Four things a replacement must keep

The mechanics around the models are load-bearing, and a rewrite that drops any
of these breaks in a way that looks like the *server's* fault:

1. **Echo `id=` back from `request.ini`.** It is what proves the result
   answers the request in flight rather than an earlier one.

2. **Write the result to a temporary name and `rename(2)` it into place.** The
   server polls for the file's existence and reads it the moment it appears. A
   result written in place is read half-formed. `rename` within a directory is
   atomic, which is what makes "the file is there" and "the file is complete"
   the same statement.

3. **Publish the pidfile only after the handlers are installed.** The server
   signals whatever pid it finds, and `sig_rul` defaults to SIGTERM, whose
   default disposition is to terminate. Publishing first means a window
   arriving in that gap kills the node instead of waking it.

4. **Do the work in the main loop, not in the handler.** The handlers here set
   a flag and return; `sigsuspend` with the stage signals blocked everywhere
   else is what closes the race between testing that flag and sleeping. A
   handler that wrote a file directly would be calling non-async-signal-safe
   functions from a signal context.

### On `sig_rul = SIGTERM`

It is the specified default and a poor choice: SIGTERM terminates by default,
so any ordinary `kill`, `systemctl stop` or shutdown is indistinguishable from
a request to run the RUL model. This works around it by taking `SIGINT` (and
`SIGQUIT`) as "stop", and its systemd unit sets `KillSignal=SIGINT` to match.

Setting

    sig_rul = SIGRTMIN

in both this `node.conf` and the server's `server.conf` removes the collision
outright, and nothing else has to change.

## Configuration

`node.conf`, read once at startup from the path in `MOTOR_AI_NODE_CONFIG`, or
`/etc/motor-ai-node/node.conf`. Every key is optional; the value in the file is
the built-in default.

Every key is one half of a contract with the server's `server.conf` — all four
have to say the same thing there. Nothing checks it, and a mismatch has no
distinctive symptom: the server signals a stage nobody is listening for and
waits out `result_timeout_ms`, once per window.

There is deliberately no window size here. The node reads whatever CSV it is
handed and reports the row count it found, so `window_rows` lives in exactly
two places rather than three.

## Building

No dependencies outside this tree — `include/` carries the feature code, the
models and the JSON parser, `kissfft/` the FFT. C++17, because `include/` uses
structured bindings; the node's own source is still plain C++:

```sh
make
```

Cross-compiling is the same rule with the variables replaced, which is all the
Yocto recipe does:

```sh
make CXX="$CXX" CXXFLAGS="$CXXFLAGS -std=c++14" LDFLAGS="$LDFLAGS"
```

## Testing it without a board

The handshake is drivable from a shell, because the server's half of it is just
"write a file, send a signal, read a file":

```sh
make
mkdir -p /tmp/md/input_data /tmp/md/results
printf 'data_dir = /tmp/md\nai_pid_file = /tmp/md/ai.pid\nmodel_root = .\n' > /tmp/node.conf

MOTOR_AI_NODE_CONFIG=/tmp/node.conf MOTOR_AI_FAKE_ANOMALY=1 ./motor_ai_node &

# stand in for motor_ai_server
cp some-window.csv /tmp/md/input_data/data.csv
printf 'id = 1\nstage = anomaly\n' > /tmp/md/results/request.ini
kill -USR1 $(cat /tmp/md/ai.pid); sleep 0.2; cat /tmp/md/results/result.ini
kill -USR2 $(cat /tmp/md/ai.pid); sleep 0.2; cat /tmp/md/results/result.ini
kill -TERM $(cat /tmp/md/ai.pid); sleep 0.2; cat /tmp/md/results/result.ini

kill -INT $(cat /tmp/md/ai.pid)     # SIGINT, not SIGTERM -- see above
```

`MOTOR_AI_FAKE_ANOMALY=1` makes the anomaly stage report an anomaly on every
window. Without it the other two stages are unreachable on a healthy rig,
because the server only asks for them when the first answers something other
than `normal`.

`some-window.csv` must be a real recording in the logger's format —
`timestamp, Current_0..2, Speed_volt_cmd, Volt_0..2, DC_bus_volt, vib_x/y/z` —
with at least `window_rows` rows. Columns the node does not know about are
ignored; the ones it needs are listed in `config/feature_extraction.json`.

Feeding the same file repeatedly does not fill the ring: the window is cached
on its identity, so the second and third stages of one request reuse the
features rather than recomputing them. Write a *different* window each time to
see the ring behave as it does in service.
