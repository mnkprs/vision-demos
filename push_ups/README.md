# push_ups

Counts push-up reps from a side-on clip, times the way down and the way up on
each one, and grades every rep's depth as **full** or **partial** from the
elbow angle at the bottom. It also follows the body line through every rep and
flags hips that sit high (piked) or sag, and where in the rep that happened.
The returned keypoints are drawn back onto the video beside a stats panel: the
per-rep numbers and graph, and a live block that moves with every frame (the
elbow angle on a dial, whether you are going down, pushing up or resting, the
rep timer, the body line, and how deep you are, from arms locked at 0% to
shoulders at the height of your hands at 100%). Pose estimation runs on
[`vitpose-plus-large`](https://vlm.run/gateway/models/usyd-community-vitpose-plus-large)
through the [VLM Run Gateway](https://www.vlm.run/gateway), so there are no
model weights to download.

For how a rep is found, timed and graded, see
[rep-counting-explained.md](rep-counting-explained.md).

## Run it

1. **Get an API key** at [app.vlm.run/sign-in](https://app.vlm.run/sign-in).

2. **Set it** in a `.env` at the repo root:

   ```bash
   cp ../.env.example ../.env
   # paste your key after VLMRUN_API_KEY=
   ```

3. **Create the conda environment:**

   ```bash
   conda env create -f environment.yml
   conda activate push_ups
   ```

4. **Film a set from the side**, with the whole body in shot: hands, shoulders,
   hips and feet. Standing up before or after the set is fine; those frames are
   ignored. Put the clip under `data/input/`.

5. **Run it** from this directory:

   ```bash
   python main.py data/input/my_set.mov             # this clip
   python main.py                                   # config.py's INPUT_VIDEO
   python main.py my_set.mov --li "Your Name" --x @yourhandle
   ```

   A few flags override their [`config.py`](config.py) value for that run only:

   | flag | what it sets |
   |---|---|
   | `CLIP` | the clip to run; without one, `INPUT_VIDEO` |
   | `--height 1080` | output video height in px, or `full`; the model still sees `INFERENCE_HEIGHT`, so no new call (`EXPORT_HEIGHT`) |
   | `--trim 5` | only the first 5 s of the clip, for a quick test (`TRIM_SECONDS`) |
   | `--li` `--x` `--ig` / `--no-credit` | the credit in the corner (`CREDIT`) |

   `python main.py -h` lists them. Everything else stays in `config.py`,
   including the full-depth cutoff (`FULL_DEPTH_ELBOW_DEG`, 90°), the body-line
   tolerance (`BODY_LINE_TOLERANCE_DEG`, 10°), what the graph plots
   (`REP_METRIC`: depth in degrees, or a time), and the live block
   (`PANEL_LIVE`, its smoothing and its sizes).

## Output

One timestamped directory per run under `data/output/`:

```
20261010-161204/
├── <clip>_pose.mp4     # H.264 overlay video + stats panel
├── displacement.png    # depth, elbow angle and body line over time, every rep marked
├── reps.json           # per-rep times and grades + the full signal series
├── summary.txt         # the rep report, human-readable
├── metrics.txt         # performance, with metrics.json beside it
├── poses.json          # the raw gateway response
└── run.json            # config + provenance
```

- Every knob lives in [`config.py`](config.py), and each run snapshots the ones
  it used into `run.json` and `reps.json`.
- Converted MP4s and gateway responses are cached in `data/cache/`, keyed on the
  source file *and* the settings, so a changed setting never reuses a stale file.
  Set `REUSE_POSES = True` to re-render or re-tune without a new gateway call.
- Both directories are gitignored, so nothing you record ends up in the repo.

## License

[Apache-2.0](../LICENSE).
