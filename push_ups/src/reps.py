"""Shoulder depth, rep counting, and each rep's depth grade and body line.

The signal is the shoulders' height above the wrists. The hands are planted on
the floor, so they are a fixed point the way the hands on the bar are for
chin-ups, and a handheld camera does not register as movement. A push-up rests
at the *top*, though, so the signal is read the other way up from a chin-up's:
zero is the top plateau and a rep is a dip below it, measured as depth. See
rep-counting-explained.md for the method, and config.py for each default.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

from .skeleton import KPT_NAMES

_INDEX = {name: i for i, name in enumerate(KPT_NAMES)}

# What each cfg.REP_METRIC reports, as (Rep attribute, name, unit, graph title, y label).
METRICS = {
    "depth": ("elbow_degrees", "elbow angle at the bottom", "°",
              "Elbow at Bottom (°)", "Elbow angle (°)"),
    "total": ("total_seconds", "rep time, down + up", "s", "Rep Time (s)", "Time (s)"),
    "down": ("down_seconds", "down time (eccentric)", "s", "Down Time (s)", "Time (s)"),
    "up": ("up_seconds", "up time (concentric)", "s", "Up Time (s)", "Time (s)"),
}

# How each metric names its best and worst rep. Lower is better for all four:
# a smaller elbow angle is a deeper rep, a shorter time a faster one.
EXTREME_WORDS = {
    "depth": ("deepest", "shallowest"),
    "total": ("fastest", "slowest"),
    "down": ("fastest", "slowest"),
    "up": ("fastest", "slowest"),
}


def rep_value(rep, metric: str) -> float:
    """The per-rep number the UI plots, per cfg.REP_METRIC."""
    return float(getattr(rep, METRICS[metric][0]))


def phase_label(metric: str) -> str:
    """Human name for what *metric* measures, e.g. "down time (eccentric)"."""
    return METRICS[metric][1]


def format_value(value: float, metric: str) -> str:
    """``68°`` for an angle, ``1.84s`` for a time, ``—`` for unreadable."""
    if not np.isfinite(value):
        return "—"
    return f"{value:.0f}°" if METRICS[metric][2] == "°" else f"{value:.2f}s"


def extremes(values, *, tol: float = 1e-6) -> tuple[int, int]:
    """Indices of the best (lowest) and worst (highest) rep, ties broken by position.

    Reps are frame-quantized, so exact ties are common rather than a corner
    case. The *earliest* rep to reach the best value is named (you got there
    before fatigue) and the *latest* to reach the worst (where the set ended
    up). Unreadable values are skipped.
    """
    seen = [(i, v) for i, v in enumerate(values) if np.isfinite(v)]
    if not seen:
        return 0, 0
    best = min(v for _, v in seen)
    worst = max(v for _, v in seen)
    best_i = next(i for i, v in seen if v <= best + tol)
    worst_i = next(i for i, v in reversed(seen) if v >= worst - tol)
    return best_i, worst_i


@dataclass
class Rep:
    """One top-to-bottom-to-top cycle, timed and graded."""

    number: int
    top_frame: int        # the highest point before the descent
    onset_frame: int      # where the descent starts, the edge of the top plateau
    bottom_frame: int
    lockout_frame: int    # where the press finishes, arms straight again
    onset_time: float
    bottom_time: float
    lockout_time: float
    down_seconds: float   # onset to bottom, the eccentric
    up_seconds: float     # bottom to lockout, the concentric
    total_seconds: float
    depth: float          # shoulder drop at the bottom, as a fraction of frame height
    elbow_degrees: float  # smallest elbow angle in the rep; NaN when unreadable
    elbow_side: str       # the arm it was read on
    # The furthest the hips stayed off the shoulder-to-feet line, anywhere in the
    # rep or the rest at the top after it: positive below the line (sagging),
    # negative above it (hips high, piked). NaN when unreadable.
    hip_degrees: float
    hip_frame: int        # where that was
    hip_time: float
    hip_where: str        # "top", "going down", "bottom" or "pushing up"
    hip_reference: str    # "ankle", or "knee" when the ankle was not clear enough
    body_side: str
    full_depth: bool
    body_line: str        # "straight", "sagging", "piked" (hips high) or "unreadable"

    @property
    def complete(self) -> bool:
        return self.full_depth

    def as_dict(self) -> dict:
        def num(value, places):
            return None if not np.isfinite(value) else round(float(value), places)

        return {
            "number": self.number,
            "onset_time": round(self.onset_time, 3),
            "bottom_time": round(self.bottom_time, 3),
            "lockout_time": round(self.lockout_time, 3),
            "down_seconds": round(self.down_seconds, 3),
            "up_seconds": round(self.up_seconds, 3),
            "total_seconds": round(self.total_seconds, 3),
            "depth": round(self.depth, 4),
            "elbow_degrees": num(self.elbow_degrees, 1),
            "elbow_side": self.elbow_side,
            "grade": "full" if self.full_depth else "partial",
            "hip_degrees": num(self.hip_degrees, 1),
            "hip_where": self.hip_where,
            "hip_frame": self.hip_frame,
            "hip_time": round(self.hip_time, 3),
            "hip_reference": self.hip_reference,
            "body_side": self.body_side,
            "body_line": self.body_line,
            "frames": {"top": self.top_frame, "onset": self.onset_frame,
                       "bottom": self.bottom_frame, "lockout": self.lockout_frame},
        }


@dataclass
class RepAnalysis:
    indices: np.ndarray        # the source frame each returned frame is
    times: np.ndarray
    raw: np.ndarray            # shoulder height above the wrists, per returned frame
    depth: np.ndarray          # smoothed, below the top plateau
    elbow: np.ndarray          # elbow angle on the clip's clearer arm, for the plot
    hip_line: np.ndarray       # signed hip angle off straight on the clearer side
    # The same, unfiltered, per foot reference ("ankle", "knee") and side, shape
    # (len(BODY_LINE_JOINTS), frames): the live readout picks each rep's own.
    hip_lines: dict
    body_sides: tuple[str, ...]
    torso_degrees: np.ndarray  # torso angle from horizontal, what the plank gate reads
    # Elbow angle per arm, unfiltered, shape (len(ARM_JOINTS), frames). The live
    # panel smooths it looking back only, where the grade's filter is centered.
    arm_raw: np.ndarray
    arm_sides: tuple[str, ...]  # "left", "right": the side each row of arm_raw is
    top: float
    amplitude: float
    full_depth_degrees: float
    body_line_tolerance: float
    elbow_side: str = ""
    hip_reference: str = ""
    reps: list[Rep] = field(default_factory=list)
    counts_by_source_frame: np.ndarray = field(default_factory=lambda: np.zeros(0, int))
    in_plank: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))

    @property
    def off_plank_frames(self) -> int:
        return int((~self.in_plank).sum()) if len(self.in_plank) else 0

    @property
    def count(self) -> int:
        return len(self.reps)

    @property
    def depth_tally(self) -> tuple[int, int]:
        """``(full-depth reps, reps)``."""
        return sum(1 for r in self.reps if r.full_depth), len(self.reps)

    def count_at(self, frame: int) -> int:
        """Reps completed by *frame*, for the progressive panel."""
        counts = self.counts_by_source_frame
        return int(counts[frame]) if frame < len(counts) else self.count

    def summary(self) -> dict:
        if not self.reps:
            return {"reps": 0}
        total = [r.total_seconds for r in self.reps]
        full, n = self.depth_tally
        return {
            "reps": n,
            "full_depth_reps": full,
            "full_depth_elbow_degrees": self.full_depth_degrees,
            "body_line_tolerance_degrees": self.body_line_tolerance,
            "body_line_flags": sum(1 for r in self.reps if r.body_line in ("sagging", "piked")),
            "top_signal": round(self.top, 4),
            "typical_depth": round(self.amplitude, 4),
            "mean_down_seconds": round(float(np.mean([r.down_seconds for r in self.reps])), 3),
            "mean_up_seconds": round(float(np.mean([r.up_seconds for r in self.reps])), 3),
            "mean_rep_seconds": round(float(np.mean(total)), 3),
            "fastest_rep_seconds": round(float(np.min(total)), 3),
            "slowest_rep_seconds": round(float(np.max(total)), 3),
            "reps_detail": [r.as_dict() for r in self.reps],
        }

    def series(self) -> dict:
        """The full per-frame signal, for reps.json."""
        def clean(values, places):
            return [None if not np.isfinite(v) else round(float(v), places) for v in values]

        return {
            "time": [round(float(v), 4) for v in self.times],
            "depth": clean(self.depth, 5),
            "elbow_degrees": clean(self.elbow, 2),
            "hip_line_degrees": clean(self.hip_line, 2),
            "torso_degrees": clean(self.torso_degrees, 2),
            "in_plank": [bool(v) for v in self.in_plank],
        }


def signal_config(cfg) -> dict:
    """The settings that shaped the signal and the grades, snapshotted into reps.json."""
    return {
        "shoulder_joints": list(cfg.SHOULDER_JOINTS),
        "wrist_joints": list(cfg.WRIST_JOINTS),
        "hip_joints": list(cfg.HIP_JOINTS),
        "min_keypoint_score": cfg.MIN_KEYPOINT_SCORE,
        "plank_max_torso_degrees": cfg.PLANK_MAX_TORSO_DEGREES,
        "plank_min_seconds": cfg.PLANK_MIN_SECONDS,
        "smooth_median_frames": cfg.SMOOTH_MEDIAN_FRAMES,
        "smooth_mean_frames": cfg.SMOOTH_MEAN_FRAMES,
        "top_percentile": cfg.TOP_PERCENTILE,
        "depth_fraction": cfg.REP_DEPTH_FRACTION,
        "reset_fraction": cfg.REP_RESET_FRACTION,
        "phase_method": cfg.REP_PHASE_METHOD,
        "full_depth_elbow_degrees": cfg.FULL_DEPTH_ELBOW_DEG,
        "body_line_tolerance_degrees": cfg.BODY_LINE_TOLERANCE_DEG,
        "body_line_hold_seconds": cfg.BODY_LINE_HOLD_SECONDS,
        "body_line_zone_fraction": cfg.BODY_LINE_ZONE_FRACTION,
        "ankle_min_score": cfg.ANKLE_MIN_SCORE,
        "grade_sample_frames": cfg.GRADE_SAMPLE_FRAMES,
    }


def _median_filter(x: np.ndarray, k: int) -> np.ndarray:
    """Kill isolated spikes without rounding off the turns. NaNs are skipped, not spread."""
    if k < 3:
        return x
    k += (k + 1) % 2   # force odd
    pad = np.pad(x, k // 2, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(pad, k)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # an all-NaN window stays NaN
        return np.nanmedian(windows, axis=1)


def _box_filter(x: np.ndarray, k: int) -> np.ndarray:
    """Moving average. Centered and edge-padded, so it does not shift the turns."""
    if k < 2:
        return x
    k += (k + 1) % 2
    pad = np.pad(x, k // 2, mode="edge")
    return np.convolve(pad, np.ones(k) / k, mode="valid")


def _fill_gaps(x: np.ndarray, limit: int | None = None) -> np.ndarray:
    """Interpolate frames where the joints were missing.

    With *limit*, only gaps of at most that many frames with a value on both
    sides are bridged; longer ones stay NaN, because nobody in frame is not the
    same as a body holding still.
    """
    valid = np.isfinite(x)
    if valid.all():
        return x
    if not valid.any():
        return x.copy() if limit is not None else np.zeros_like(x)
    idx = np.arange(len(x))
    filled = np.interp(idx, idx[valid], x[valid])
    if limit is None:
        return filled

    keep = valid.copy()
    i = 0
    while i < len(x):
        if valid[i]:
            i += 1
            continue
        j = i
        while j < len(x) and not valid[j]:
            j += 1
        if i > 0 and j < len(x) and j - i <= limit:
            keep[i:j] = True
        i = j
    return np.where(keep, filled, np.nan)


def keypoints(frames: list[dict], aspect: float) -> tuple[np.ndarray, np.ndarray]:
    """Every joint of the first tracked person, as ``(xy, score)``.

    ``xy`` is ``(frames, joints, 2)`` in units of frame height: x is scaled by
    *aspect* so angles are not distorted on a non-square frame. (0, 0) is the
    model's "not visible" sentinel, so it becomes NaN with a score of 0 rather
    than a real point in the top-left corner.
    """
    xy = np.full((len(frames), len(KPT_NAMES), 2), np.nan)
    score = np.zeros((len(frames), len(KPT_NAMES)))
    for i, frame in enumerate(frames):
        persons = frame.get("persons") or []
        if not persons:
            continue
        kpts = persons[0].get("kpts_xy") or []
        scores = persons[0].get("kpts_score") or [1.0] * len(kpts)
        for j, point in enumerate(kpts[:len(KPT_NAMES)]):
            x, y = point[0], point[1]
            if (x, y) == (0, 0):
                continue
            xy[i, j] = (x * aspect, y)
            score[i, j] = scores[j] if j < len(scores) else 1.0
    return xy, score


def weighted_point(xy: np.ndarray, score: np.ndarray, names, min_score: float) -> np.ndarray:
    """Confidence-weighted mean of the named joints per frame, ``(frames, 2)``.

    From the side the far shoulder and wrist are the model's guesses, so the
    near side, which it is surer of, carries the point. Joints under
    *min_score* are left out; NaN where none qualify.
    """
    idx = [_INDEX[n] for n in names]
    pts, w = xy[:, idx], score[:, idx]
    w = np.where((w >= min_score) & np.isfinite(pts[..., 1]), w, 0.0)
    total = w.sum(axis=1)
    mean = (np.nan_to_num(pts) * w[..., None]).sum(axis=1) / np.maximum(total, 1e-9)[:, None]
    return np.where(total[:, None] > 0, mean, np.nan)


def _angle_at(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Degrees at *b*, between b->a and b->c, per frame. NaN where a joint is unseen."""
    u, v = a - b, c - b
    nu, nv = np.linalg.norm(u, axis=-1), np.linalg.norm(v, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = np.sum(u * v, axis=-1) / (nu * nv)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def _line_deviation(shoulder: np.ndarray, hip: np.ndarray, foot: np.ndarray) -> np.ndarray:
    """Signed degrees the hip sits off the straight shoulder-to-foot line.

    Positive with the hip below the line (toward the floor, sagging), negative
    above it (piked). "Below" is read from image y, so it holds whichever way
    the athlete faces.
    """
    bend = 180.0 - _angle_at(shoulder, hip, foot)
    d = foot - shoulder
    normal = np.stack([-d[..., 1], d[..., 0]], axis=-1)
    normal = np.where(normal[..., 1:2] < 0, -normal, normal)   # point it down the image
    side = np.sum((hip - shoulder) * normal, axis=-1)
    return np.where(side >= 0, bend, -bend)


def _side_name(joints) -> str:
    return joints[0].split("_")[0]


def _surest_side(score: np.ndarray, sides, lo: int, hi: int) -> int:
    """Index of the side whose joints the model was surest of across frames lo..hi."""
    means = [score[lo:hi + 1][:, [_INDEX[n] for n in joints]].mean() for joints in sides]
    return int(np.argmax(means))


def _window_median(series: np.ndarray, center: int, width: int) -> float:
    """Median of *series* in a window centered on *center*, ignoring NaNs.

    So one bad pose at the bottom cannot decide a grade.
    """
    half = max(0, int(width) // 2)
    window = series[max(0, center - half):min(len(series), center + half + 1)]
    window = window[np.isfinite(window)]
    return float(np.median(window)) if len(window) else np.nan


def plank_mask(torso_degrees: np.ndarray, rate: float, *, cfg) -> np.ndarray:
    """True where the body is in a plank: torso near horizontal, for long enough.

    Standing, kneeling up or walking leaves the torso near vertical, so those
    frames drop out however the shoulders and wrists happen to line up.
    """
    angle = _median_filter(torso_degrees, cfg.SMOOTH_MEDIAN_FRAMES)
    mask = np.isfinite(angle) & (angle <= cfg.PLANK_MAX_TORSO_DEGREES)

    shortest = max(1, int(round(cfg.PLANK_MIN_SECONDS * rate)))
    i = 0
    while i < len(mask):
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < len(mask) and mask[j]:
            j += 1
        if j - i < shortest:
            mask[i:j] = False
        i = j
    return mask


def _phase_start(depth, top_i: int, bottom_i: int, *, amplitude: float, cfg) -> int:
    """Where the descent begins: the edge of the top plateau, not its highest point.

    ``threshold`` reads the signal at the top, where the body is nearly still
    and the noise floor is the whole signal. ``velocity`` walks back from the
    fastest point of the descent instead, reading the signal where the movement
    is large, which is what makes it steadier exactly where the threshold is not.
    """
    if bottom_i <= top_i:
        return top_i

    if cfg.REP_PHASE_METHOD == "threshold":
        sink = np.where(
            depth[top_i:bottom_i + 1] > depth[top_i] + cfg.REP_PHASE_FRACTION * amplitude
        )[0]
        return top_i + int(sink[0]) if len(sink) else top_i

    segment = depth[top_i:bottom_i + 1]
    if len(segment) < 3:
        return top_i
    velocity = np.gradient(segment)
    fastest = int(np.argmax(velocity))
    if velocity[fastest] <= 0:
        return top_i

    floor = cfg.REP_PHASE_VELOCITY_FRACTION * velocity[fastest]
    i = fastest
    while i > 0 and velocity[i] > floor:
        i -= 1
    return top_i + i


def _phase_end(depth, bottom_i: int, top_i: int, *, amplitude: float, cfg) -> int:
    """Where the press finishes: the same rule as `_phase_start`, run backwards in time.

    Reversed, the press is a descent from the top that follows it, so one
    implementation finds both ends and the two cannot disagree on method.
    """
    if top_i <= bottom_i:
        return bottom_i
    last = len(depth) - 1
    start = _phase_start(depth[::-1], last - top_i, last - bottom_i, amplitude=amplitude, cfg=cfg)
    return last - start


def _robust_amplitude(depth: np.ndarray, valid: np.ndarray) -> float:
    """Typical rep depth, as the median of the bottoms rather than the deepest.

    ``max()`` is one frame's opinion, and every threshold downstream is a
    fraction of this number, so one bad pose would move all of them.
    """
    usable = depth[valid]
    if not len(usable):
        return 0.0
    deepest = float(np.max(usable))
    if deepest <= 0:
        return 0.0

    bottoms = [
        depth[i]
        for i in range(1, len(depth) - 1)
        if valid[i]
        and depth[i] >= depth[i - 1]
        and depth[i] > depth[i + 1]
        and depth[i] > 0.3 * deepest
    ]
    if len(bottoms) >= 3:
        return float(np.median(bottoms))
    return float(np.percentile(usable, 95))


def analyze(frames: list[dict], fps: float, n_source_frames: int, *, cfg,
            aspect: float) -> RepAnalysis:
    """Build the depth signal, count reps off it, and grade each one."""
    indices = np.array([f["index"] for f in frames], dtype=int)
    times = indices / fps
    # Returned frames per second; equal to fps unless the model skipped frames.
    rate = fps * len(frames) / max(n_source_frames, 1)
    xy, score = keypoints(frames, aspect)

    shoulders = weighted_point(xy, score, cfg.SHOULDER_JOINTS, cfg.MIN_KEYPOINT_SCORE)
    wrists = weighted_point(xy, score, cfg.WRIST_JOINTS, cfg.MIN_KEYPOINT_SCORE)
    hips = weighted_point(xy, score, cfg.HIP_JOINTS, cfg.MIN_KEYPOINT_SCORE)

    # Image y grows downward, so this is largest at the top of a push-up and
    # shrinks as the chest drops toward the hands.
    raw = _fill_gaps(wrists[:, 1] - shoulders[:, 1])

    # 0 is a horizontal torso, 90 an upright one. A dropout shorter than the
    # plank minimum is bridged; a longer one means nobody is there.
    torso_vec = shoulders - hips
    torso = np.degrees(np.arctan2(np.abs(torso_vec[:, 1]), np.abs(torso_vec[:, 0])))
    torso = _fill_gaps(torso, limit=int(round(cfg.PLANK_MIN_SECONDS * rate)))
    in_plank = plank_mask(torso, rate, cfg=cfg)

    smooth = _box_filter(_median_filter(raw, cfg.SMOOTH_MEDIAN_FRAMES), cfg.SMOOTH_MEAN_FRAMES)

    # Relative zero is the top plateau, taken as a high percentile rather than
    # the maximum so one bad frame cannot define it. Depth is measured down
    # from there, which turns the dip into a rise the counter reads directly.
    usable = smooth[in_plank] if in_plank.any() else smooth
    top = float(np.percentile(usable, cfg.TOP_PERCENTILE))
    depth = top - smooth
    amplitude = _robust_amplitude(depth, in_plank)

    def joints(*names):
        return [xy[:, _INDEX[n]] for n in names]

    arm_raw = np.stack([_angle_at(*joints(*arm)) for arm in cfg.ARM_JOINTS])
    arms = np.stack([_median_filter(a, cfg.GRADE_SAMPLE_FRAMES) for a in arm_raw])
    lines = {
        "ankle": np.stack([_line_deviation(*joints(s, h, a))
                           for s, h, _, a in cfg.BODY_LINE_JOINTS]),
        "knee": np.stack([_line_deviation(*joints(s, h, k))
                          for s, h, k, _ in cfg.BODY_LINE_JOINTS]),
    }

    def foot_reference(side: int, lo: int, hi: int) -> str:
        """Ankle if the model saw it clearly enough across lo..hi, else knee."""
        ankle = _INDEX[cfg.BODY_LINE_JOINTS[side][3]]
        return "ankle" if score[lo:hi + 1, ankle].mean() >= cfg.ANKLE_MIN_SCORE else "knee"

    # The clip's clearer arm and side, for the per-frame plot. Each rep picks its own.
    span = (0, len(frames) - 1)
    if in_plank.any():
        where = np.flatnonzero(in_plank)
        span = (int(where[0]), int(where[-1]))
    arm_side = _surest_side(score, cfg.ARM_JOINTS, *span)
    body_side = _surest_side(score, cfg.BODY_LINE_JOINTS, *span)
    clip_ref = foot_reference(body_side, *span)

    spans = _count(depth, amplitude, in_plank, cfg=cfg)
    run_end = _run_ends(in_plank)

    phases = []
    for k, (top_i, bottom_i, close_i) in enumerate(spans):
        # The top after this rep is the next one's top while the plank holds,
        # else the highest point before the plank ends.
        if k + 1 < len(spans) and spans[k + 1][0] <= run_end[close_i]:
            after_i = spans[k + 1][0]
        else:
            stretch = depth[close_i:run_end[close_i] + 1]
            after_i = close_i + int(np.argmin(stretch))
        onset_i = _phase_start(depth, top_i, bottom_i, amplitude=amplitude, cfg=cfg)
        lockout_i = _phase_end(depth, bottom_i, after_i, amplitude=amplitude, cfg=cfg)
        phases.append((top_i, onset_i, bottom_i, lockout_i, close_i))

    hold = max(1, int(round(cfg.BODY_LINE_HOLD_SECONDS * rate)))
    reps: list[Rep] = []
    for k, (top_i, onset_i, bottom_i, lockout_i, close_i) in enumerate(phases):
        # The body line is read across the whole rep and the rest at the top
        # after it: up to the next rep's onset while the plank holds, else to
        # the end of the plank.
        if k + 1 < len(phases) and phases[k + 1][1] <= run_end[close_i]:
            end_i = phases[k + 1][1]
        else:
            end_i = run_end[close_i] + 1

        arm = _surest_side(score, cfg.ARM_JOINTS, onset_i, lockout_i)
        window = arms[arm, onset_i:lockout_i + 1]
        # To the whole degree, the precision the panel shows and more than a 2D
        # angle really has, so a rep shown at 90° is never graded on 90.4°.
        elbow = float(np.round(np.nanmin(window))) if np.isfinite(window).any() else np.nan

        side = _surest_side(score, cfg.BODY_LINE_JOINTS, onset_i, end_i - 1)
        ref = foot_reference(side, onset_i, end_i - 1)
        line = _median_filter(lines[ref][side], cfg.GRADE_SAMPLE_FRAMES)
        hip, at = _held_extreme(line[onset_i:end_i], hold)
        hip_i = onset_i + at
        where = _where(hip_i, onset_i, bottom_i, lockout_i, depth, cfg=cfg)

        reps.append(_make_rep(
            len(reps) + 1, top_i, onset_i, bottom_i, lockout_i, times, indices, depth,
            elbow=elbow, elbow_side=_side_name(cfg.ARM_JOINTS[arm]),
            hip=hip, hip_frame=int(indices[hip_i]), hip_time=float(times[hip_i]),
            hip_where=where, hip_reference=ref,
            body_side=_side_name(cfg.BODY_LINE_JOINTS[side]), cfg=cfg))

    counts = np.zeros(max(n_source_frames, 1), dtype=int)
    for rep in reps:
        if rep.lockout_frame < len(counts):
            counts[rep.lockout_frame:] = rep.number
    return RepAnalysis(
        indices=indices, times=times, raw=raw, depth=depth, elbow=arms[arm_side],
        hip_line=lines[clip_ref][body_side], hip_lines=lines,
        body_sides=tuple(_side_name(s) for s in cfg.BODY_LINE_JOINTS),
        torso_degrees=torso, arm_raw=arm_raw,
        arm_sides=tuple(_side_name(arm) for arm in cfg.ARM_JOINTS), top=top,
        amplitude=amplitude, full_depth_degrees=cfg.FULL_DEPTH_ELBOW_DEG,
        body_line_tolerance=cfg.BODY_LINE_TOLERANCE_DEG,
        elbow_side=_side_name(cfg.ARM_JOINTS[arm_side]), hip_reference=clip_ref,
        reps=reps, counts_by_source_frame=counts, in_plank=in_plank)


def _held_extreme(line: np.ndarray, hold: int) -> tuple[float, int]:
    """The furthest the hips stayed off the line for *hold* frames running, and where.

    Signed like the line: + sagging, - hips high. For each frame, the run of
    *hold* frames starting there is only as far off as its closest frame, so a
    one-frame spike cannot set the verdict. Returns ``(degrees, index)`` with the
    index at the middle of that run; NaN when the line was never readable.
    """
    best, at = np.nan, 0
    if len(line) < hold or not np.isfinite(line).any():
        return best, at
    runs = np.lib.stride_tricks.sliding_window_view(line, hold)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # an all-NaN run stays NaN
        for sign in (1.0, -1.0):
            held = np.nanmin(sign * runs, axis=1)
            if not np.isfinite(held).any():
                continue
            j = int(np.nanargmax(held))
            if not np.isfinite(best) or held[j] > abs(best):
                best, at = sign * max(float(held[j]), 0.0), j + hold // 2
    return best, at


def _where(i: int, onset_i: int, bottom_i: int, lockout_i: int, depth, *, cfg) -> str:
    """Which part of the rep frame *i* is: "top", "going down", "bottom" or "pushing up".

    Read off the depth rather than the clock, so a long pause at the bottom
    is all "bottom" and the last stretch of the press is already "top".
    """
    deepest = depth[bottom_i]
    if deepest > 0:
        share = depth[i] / deepest
        if share >= 1 - cfg.BODY_LINE_ZONE_FRACTION:
            return "bottom"
        if share <= cfg.BODY_LINE_ZONE_FRACTION or i >= lockout_i:
            return "top"
    if i < onset_i:
        return "top"
    return "going down" if i < bottom_i else "pushing up"


def _run_ends(mask: np.ndarray) -> np.ndarray:
    """For each frame in a True run, the index of that run's last frame."""
    out = np.zeros(len(mask), dtype=int)
    end = len(mask) - 1
    for i in range(len(mask) - 1, -1, -1):
        if not mask[i]:
            end = i - 1
        out[i] = end
    return out


def _count(depth, amplitude, in_plank, *, cfg) -> list[tuple[int, int, int]]:
    """Hysteresis state machine: ``(top, bottom, close)`` indices per rep.

    Armed at the top, confirmed once the shoulders sink past a share of a
    typical rep, closed when they come back near the top. It starts disarmed,
    and disarms whenever the plank breaks, so every rep has a top seen before
    it inside one unbroken plank: a clip that opens at the bottom does not count
    its first press (the descent was never seen), and dropping to the knees at
    the bottom cannot be finished by a later plank.
    """
    if amplitude <= 0 or len(depth) < 3:
        return []

    high = cfg.REP_DEPTH_FRACTION * amplitude
    low = cfg.REP_RESET_FRACTION * amplitude

    spans: list[tuple[int, int, int]] = []
    state, top_i, bottom_i = "off", 0, 0
    for i, d in enumerate(depth):
        if not in_plank[i]:
            state = "off"   # out of the plank: an open rep is dropped, not finished
            continue
        if state == "off":
            if d < low:
                state, top_i = "top", i
        elif state == "top":
            if d < depth[top_i]:
                top_i = i
            if d > high:
                state, bottom_i = "bottom", i
        else:
            if d > depth[bottom_i]:
                bottom_i = i
            if d < low:
                spans.append((top_i, bottom_i, i))
                state, top_i = "top", i
    # A rep still at the bottom when the clip or the plank ends was never
    # pressed back up, so it is not a push-up.
    return spans


def _make_rep(number, top_i, onset_i, bottom_i, lockout_i, times, indices, depth, *,
              elbow, elbow_side, hip, hip_frame, hip_time, hip_where, hip_reference,
              body_side, cfg) -> Rep:
    tol = cfg.BODY_LINE_TOLERANCE_DEG
    if not np.isfinite(hip):
        line = "unreadable"
    elif abs(hip) <= tol:
        line = "straight"
    else:
        line = "sagging" if hip > 0 else "piked"
    return Rep(
        number=number,
        top_frame=int(indices[top_i]),
        onset_frame=int(indices[onset_i]),
        bottom_frame=int(indices[bottom_i]),
        lockout_frame=int(indices[lockout_i]),
        onset_time=float(times[onset_i]),
        bottom_time=float(times[bottom_i]),
        lockout_time=float(times[lockout_i]),
        down_seconds=float(times[bottom_i] - times[onset_i]),
        up_seconds=float(times[lockout_i] - times[bottom_i]),
        total_seconds=float(times[lockout_i] - times[onset_i]),
        depth=float(depth[bottom_i]),
        elbow_degrees=elbow,
        elbow_side=elbow_side,
        hip_degrees=hip,
        hip_frame=hip_frame,
        hip_time=hip_time,
        hip_where=hip_where,
        hip_reference=hip_reference,
        body_side=body_side,
        # Unreadable is not passing: assuming depth would flatter the athlete.
        full_depth=bool(np.isfinite(elbow) and elbow <= cfg.FULL_DEPTH_ELBOW_DEG),
        body_line=line,
    )


# How each body-line verdict reads to a person.
BODY_LINE_LABELS = {
    "straight": "straight",
    "piked": "hips high",
    "sagging": "hips sagging",
    "unreadable": "not readable",
}


def _ranges(numbers: list[int]) -> str:
    """``[1, 2, 3, 5]`` -> ``"1–3, 5"``."""
    parts, start = [], None
    for k, n in enumerate(numbers):
        start = n if start is None else start
        if k + 1 == len(numbers) or numbers[k + 1] != n + 1:
            parts.append(str(n) if n == start else f"{start}–{n}")
            start = None
    return ", ".join(parts)


def body_line_note(reps, *, compact: bool = False) -> str:
    """One line on the body line across the set, naming the reps that broke it.

    "Hips high on reps 1–5, worst 13° at the top". *compact* is the shorter form
    the panel's closing note has room for.
    """
    if not reps:
        return ""
    read = [r for r in reps if r.body_line != "unreadable"]
    if not read:
        return "Body line: not readable"
    flagged = [r for r in read if r.body_line in ("sagging", "piked")]
    if not flagged:
        return "Body line: straight on every rep" if len(read) == len(reps) \
            else "Body line: straight where readable"

    worst = max(flagged, key=lambda r: abs(r.hip_degrees))
    groups = []
    for verdict in ("piked", "sagging"):
        numbers = [r.number for r in flagged if r.body_line == verdict]
        if numbers:
            label = "rep" if len(numbers) == 1 else "reps"
            groups.append((BODY_LINE_LABELS[verdict].capitalize(), label, _ranges(numbers)))
    if len(groups) == 1:
        word, label, numbers = groups[0]
        if compact:
            return f"{word}, {label} {numbers} ({abs(worst.hip_degrees):.0f}° at {worst.hip_where})"
        return (f"{word} on {label} {numbers}, worst {abs(worst.hip_degrees):.0f}° "
                f"at the {worst.hip_where}")
    (w1, l1, n1), (w2, _, n2) = groups
    return f"{w1} on {l1} {n1}; {w2.lower()} on {n2}"


def summary_text(analysis: RepAnalysis, *, video_seconds: float, source: str = "",
                 metric: str = "depth") -> str:
    """Human-readable rep report, saved alongside the machine-readable JSON."""
    lines = ["Push-up summary", "=" * 72]
    if source:
        lines.append(f"{'source':<22}{source}")
    lines += [
        f"{'clip length':<22}{video_seconds:.2f}s",
        f"{'total reps':<22}{analysis.count}",
    ]

    if not analysis.reps:
        lines.append("\nNo reps detected. Check displacement.png.")
        return "\n".join(lines) + "\n"

    s = analysis.summary()
    full, n = analysis.depth_tally
    values = [rep_value(r, metric) for r in analysis.reps]
    best_i, worst_i = extremes(values)
    best_word, worst_word = EXTREME_WORDS[metric]
    totals = [r.total_seconds for r in analysis.reps]
    work = float(np.sum(totals))

    lines += [
        f"{'full depth':<22}{full} of {n}   (elbow at or under "
        f"{analysis.full_depth_degrees:.0f}°)",
        f"{'body line':<22}{body_line_note(analysis.reps).removeprefix('Body line: ')}"
        f"   (off the line by more than {analysis.body_line_tolerance:.0f}°)",
        "",
        f"{'reported':<22}{phase_label(metric)}",
        f"{best_word:<22}{format_value(values[best_i], metric)}   (rep {best_i + 1})",
        f"{worst_word:<22}{format_value(values[worst_i], metric)}   (rep {worst_i + 1})",
        "",
        f"{'mean down':<22}{s['mean_down_seconds']:.2f}s   (onset to bottom)",
        f"{'mean up':<22}{s['mean_up_seconds']:.2f}s   (bottom to lockout)",
        f"{'mean rep':<22}{s['mean_rep_seconds']:.2f}s",
        f"{'time under tension':<22}{work:.2f}s of {video_seconds:.2f}s "
        f"({work / video_seconds * 100:.0f}%)",
        f"{'pace':<22}{analysis.count / video_seconds * 60:.1f} reps/min",
        "",
        "Per rep",
        "-" * 72,
        f"{'rep':>3} {'onset':>7} {'bottom':>7} {'lockout':>7} {'down':>6} {'up':>6} "
        f"{'total':>6} {'elbow':>6} {'depth':>8} {'hip':>6}  body line (worst, where)",
    ]
    for r in analysis.reps:
        hip = "—" if not np.isfinite(r.hip_degrees) else f"{r.hip_degrees:+.0f}°"
        lines.append(
            f"{r.number:>3} {r.onset_time:>6.2f}s {r.bottom_time:>6.2f}s "
            f"{r.lockout_time:>6.2f}s {r.down_seconds:>5.2f}s {r.up_seconds:>5.2f}s "
            f"{r.total_seconds:>5.2f}s {format_value(r.elbow_degrees, 'depth'):>6} "
            f"{'full' if r.full_depth else 'partial':>8} {hip:>6}  "
            f"{BODY_LINE_LABELS[r.body_line]}"
            + (f" at the {r.hip_where}, to the {r.hip_reference}"
               if np.isfinite(r.hip_degrees) else "")
        )

    rests = [
        analysis.reps[i + 1].onset_time - analysis.reps[i].lockout_time
        for i in range(len(analysis.reps) - 1)
    ]
    if rests:
        lines += [
            "",
            "Rest at the top between reps",
            "-" * 72,
            "  " + "  ".join(f"{r:.2f}s" for r in rests)
            + f"   (mean {float(np.mean(rests)):.2f}s)",
        ]

    lines += [
        "",
        "`elbow` is the smallest shoulder-elbow-wrist angle in the rep, on the arm",
        "the model saw best. `hip` is the furthest the hips stayed off the",
        "shoulder-to-feet line for a moment, anywhere in the rep or the rest at the",
        "top after it: + below the line (sagging), - above it (hips high).",
    ]
    return "\n".join(lines) + "\n"
