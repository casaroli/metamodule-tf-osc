---
name: tfexample-run-sim
description: Launch the MetaModule GUI simulator in the background so the user can interactively play TFOsc. Use when the user wants to "open the simulator", "play with it", "see the module", etc.
---

# Launch the GUI simulator

The simulator binary opens an SDL window showing the MetaModule's
240×320 display, an audio device for the soundcard, and keyboard
bindings for the encoder + buttons + knobs.

## Prerequisites

The Default (GUI) preset must be built. If currently on `headless-min`,
switch first via the `tfexample-build-sim` skill with arg `gui`.

## Run

```bash
cd /Users/marco/ia/metamodule/metamodule/simulator
pkill -9 -f "build/simulator" 2>/dev/null
sleep 1
rm -f /tmp/sim-run.log
script -F -q /tmp/sim-run.log ./build/simulator --assets build/assets.uimg --zoom 200
```

Launch in **background** (`run_in_background: true` on the Bash tool)
so the user can interact with the SDL window. Use `script -F -q` to
get line-flushed stdout (otherwise the sim's stdio buffers and the log
appears empty for ~minutes).

Then wait for readiness with:

```bash
until grep -q "buffers have # frames" /tmp/sim-run.log 2>/dev/null; do sleep 1; done
echo "sim ready"
```

## Tell the user how to navigate

Always include this cheat sheet:

| Key | Action |
|---|---|
| `1` | Lock audio routing to Out 1 → L, Out 2 → R (do FIRST) |
| `↑` | Back / aux button |
| `←` `→` | Encoder rotate |
| `↓` | Encoder press |
| `a`–`f` | Select knob A–F (Freq, Fine, Morph X, Morph Y, Warp, FM Amt) |
| `u`–`z` | Select knob u-z if more than 6 knobs (LFO Rate, LFO Depth) |
| `[` `]` | Decrease / increase selected knob by 5% |

**Workflow to actually hear TFOsc**:
1. Press `1` first (audio routing)
2. Mash `↑` until you see the patch selector
3. Encoder to `tfosc_test.yml`, press encoder to open
4. Once in patch view: turn encoder until **Play** button is highlighted
5. Press encoder to start playback
6. Now select knobs with letter keys and turn with `[` / `]`

## When sim closes

The user clicking the SDL window's close button or pressing Escape
ends the background process. The `bash` task notification will fire.
Don't auto-relaunch — wait for the user to react ("how did it sound?").

## Don't sleep-poll the log

If you need to check sim output, use the harness's bash `run_in_background`
pattern with an `until grep` loop, not a manual sleep chain.
