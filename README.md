# motor_ai_node

The inference half of the QNX motor demo's AI pipeline. It runs on the Linux
guest (guest-2), beside [motor_ai_server][server], and sleeps until that
service asks it for a verdict.

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
    /motor_data/results/result.ini     one stage's verdict, written by this
    /motor_data/ai.pid                 this process's pid, for the server to signal

## The protocol

| signal (default) | stage                 | `stage=` written | asked for when     |
|------------------|-----------------------|------------------|--------------------|
| `SIGUSR1`        | anomaly detection     | `anomaly`        | every window       |
| `SIGUSR2`        | fault classification  | `fault_class`    | anomaly != normal  |
| `SIGTERM`        | remaining useful life | `rul`            | anomaly != normal  |

The server sends one signal, waits for `results/result.ini` to appear, checks
its `stage=` matches what it asked for, and takes `value=`. A healthy motor
costs one model; only a suspect one pays for all three.

`result.ini` is the same `key = value` syntax as the config files:

```ini
# written by motor_ai_node
stage = anomaly
value = normal
rows = 26000
```

### Three things a replacement must keep

The models here are placeholders. Swapping them for real inference is meant to
be a change to `runAnomaly()` and its two siblings and nothing else — but the
mechanics around them are load-bearing, and a rewrite that drops any of these
breaks in a way that looks like the *server's* fault:

1. **Write the result to a temporary name and `rename(2)` it into place.** The
   server polls for the file's existence and reads it the moment it appears. A
   result written in place is read half-formed. `rename` within a directory is
   atomic, which is what makes "the file is there" and "the file is complete"
   the same statement.

2. **Publish the pidfile only after the handlers are installed.** The server
   signals whatever pid it finds, and `sig_rul` defaults to SIGTERM, whose
   default disposition is to terminate. Publishing first means a window
   arriving in that gap kills the node instead of waking it.

3. **Do the work in the main loop, not in the handler.** The handlers here set
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

No dependencies. Any C++14 compiler:

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
printf 'data_dir = /tmp/md\nai_pid_file = /tmp/md/ai.pid\n' > /tmp/node.conf

MOTOR_AI_NODE_CONFIG=/tmp/node.conf MOTOR_AI_FAKE_ANOMALY=1 ./motor_ai_node &

# stand in for motor_ai_server
cp some-window.csv /tmp/md/input_data/data.csv
kill -USR1 $(cat /tmp/md/ai.pid); sleep 0.2; cat /tmp/md/results/result.ini
kill -USR2 $(cat /tmp/md/ai.pid); sleep 0.2; cat /tmp/md/results/result.ini
kill -TERM $(cat /tmp/md/ai.pid); sleep 0.2; cat /tmp/md/results/result.ini

kill -INT $(cat /tmp/md/ai.pid)     # SIGINT, not SIGTERM -- see above
```

`MOTOR_AI_FAKE_ANOMALY=1` makes the anomaly stage report an anomaly on every
window. Without it the other two stages are unreachable, because the server
only asks for them when the first answers something other than `normal`.
