"""The export video, the timeline plot and the text summary."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn

from .comet import Comet
from .panel import Panel, display_smooth
from .reps import hinge_points, screen_angle
from .skeleton import KPT_NAMES, draw_person, visible_parts
from .text import resolve_font
from .video import VideoInfo

_PROGRESS_COLUMNS = (
    TextColumn("[cyan]rendering[/]"),
    BarColumn(),
    TaskProgressColumn(),
    TextColumn("{task.completed}/{task.total} frames"),
    TimeElapsedColumn(),
)


def _draw_plate(frame, plate, i: int, *, cfg, stroke: int, fps: float, comet) -> None:
    """The plate's outline, lightly filled, and a fading comet behind its center."""
    if plate is None:
        return
    h, w = frame.shape[:2]
    outline = plate.outline_at(i)
    if outline is not None and np.isfinite(outline).all():
        pts = (outline * [w, h]).astype(np.int32).reshape(-1, 1, 2)
        layer = frame.copy()
        cv2.fillPoly(layer, [pts], cfg.PLATE_COLOR)
        cv2.addWeighted(layer, cfg.PLATE_FILL_OPACITY, frame, 1 - cfg.PLATE_FILL_OPACITY, 0,
                        dst=frame)
        cv2.polylines(frame, [pts], True, cfg.PLATE_COLOR, stroke, cv2.LINE_AA)

    if comet is not None and np.isfinite(plate.per_frame_cx[i]):
        # The last BAR_PATH_SECONDS of the plate's center, oldest first; the
        # comet tapers and fades it toward the tail. No wind: the bar moves
        # straight up and down, and a drift would misstate where it went.
        a = max(0, i - int(cfg.BAR_PATH_SECONDS * fps))
        xs, ys = plate.per_frame_cx[a:i + 1], plate.per_frame_cy[a:i + 1]
        path = [(int(x * w), int(y * h)) if np.isfinite(x) and np.isfinite(y) else None
                for x, y in zip(xs, ys)]
        comet.draw(frame, [(path, cfg.BAR_TRAIL_COLOR)])
        # The head, ringed in the plate's own blue so it reads against the hub.
        head = (int(plate.per_frame_cx[i] * w), int(plate.per_frame_cy[i] * h))
        r = max(3, stroke * 2)
        cv2.circle(frame, head, r, cfg.BAR_TRAIL_COLOR, -1, cv2.LINE_AA)
        cv2.circle(frame, head, r, cfg.PLATE_COLOR, max(1, stroke // 2), cv2.LINE_AA)


def _label_width(font, px: int, text: str) -> float:
    """Rendered width of *text*, for placing the hinge label by its edge rather than its center."""
    from PIL import ImageFont

    return ImageFont.truetype(font[0], px, index=font[1]).getlength(text)


def _draw_hinge(frame, person, angle: float, *, side: str, facing: int, clockwise: bool,
                cfg, scale: float, font) -> None:
    """The angle at the hip, as an arc between thigh and torso, and its value.

    No lines of its own: the skeleton already draws shoulder-hip and hip-knee,
    and those are the angle's two sides.

    The number is the smoothed hip angle the fallback counts reps on, so the
    overlay shows the same signal the timeline plots. Side and placement are
    fixed for the clip: the arc is always on the same hip, and the value always
    sits level with it, behind the lifter, where the frame is background rather
    than arms and bar.
    """
    h, w = frame.shape[:2]
    pts = hinge_points(person, side=side, width=w, height=h, min_score=cfg.HINGE_MIN_SCORE)
    if pts is None or not np.isfinite(angle):
        return
    shoulder, hip, knee = pts
    color = cfg.HINGE_COLOR
    stroke = max(2, round(cfg.LINE_THICKNESS * scale) + 1)
    c = tuple(int(v) for v in hip)

    # The arc between thigh and torso, always opening the same way round: the
    # way it opens when bent over. Past a straight line (a slight lean back at
    # lockout) it stops at a half-circle rather than swinging to the other side.
    a_sh, a_kn = screen_angle(shoulder - hip), screen_angle(knee - hip)
    if clockwise:
        start, sweep = a_kn, (a_sh - a_kn) % 360
    else:
        start, sweep = a_sh, (a_kn - a_sh) % 360
    sweep = min(sweep, 180.0)
    r = int(cfg.HINGE_ARC_RADIUS * scale)
    layer = frame.copy()
    cv2.ellipse(layer, c, (r, r), 0, start, start + sweep, color, -1, cv2.LINE_AA)
    cv2.addWeighted(layer, 0.25, frame, 0.75, 0, dst=frame)
    cv2.ellipse(frame, c, (r, r), 0, start, start + sweep, color, max(1, stroke // 2),
                cv2.LINE_AA)

    # The value, just above the hip on the side the lifter faces away from: its
    # near corner sits a little past the arc, diagonally up and back from the
    # joint, so it reads as belonging to that point.
    if cfg.HINGE_LABEL == "none":
        return
    text = f"hip {angle:.0f}°" if cfg.HINGE_LABEL == "hip" else f"{angle:.0f}°"
    px = int(cfg.HINGE_LABEL_SIZE * scale)
    half_w = (font and _label_width(font, px, text) or px * 0.6 * len(text)) / 2
    gap = r * 0.6
    at = hip + np.array([-facing * (gap + half_w), -(gap + px * 0.5)])
    if font is None:
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, px / 34, 2)
        org = (int(at[0] - tw / 2), int(at[1] + th / 2))
        cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, px / 34, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, px / 34, (255, 255, 255), 2,
                    cv2.LINE_AA)
        return
    from PIL import Image, ImageDraw, ImageFont

    # Only the label's own patch goes through Pillow, not the whole frame.
    face = ImageFont.truetype(font[0], px, index=font[1])
    pad = px
    x0, y0 = int(at[0]) - 2 * px - pad, int(at[1]) - px - pad
    x1, y1 = int(at[0]) + 2 * px + pad, int(at[1]) + px + pad
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    if x1 <= x0 or y1 <= y0:
        return
    patch = Image.fromarray(frame[y0:y1, x0:x1, ::-1])
    ImageDraw.Draw(patch).text((at[0] - x0, at[1] - y0), text, font=face, anchor="mm",
                               fill=(255, 255, 255), stroke_width=max(2, px // 10),
                               stroke_fill=(20, 22, 26))
    frame[y0:y1, x0:x1] = np.array(patch)[..., ::-1]


def _person_at(by_index: dict, i: int, hold: int):
    """This frame's lifter, or the last one seen within *hold* frames."""
    for j in range(i, max(-1, i - hold - 1), -1):
        persons = by_index.get(j)
        if persons:
            return persons[0]
    return None


def render(info: VideoInfo, *, by_index, plate, signal, analysis, crops: dict, dst: Path,
           cfg, console) -> dict:
    """Write the clip, overlaid, beside its panel: same height, double the width.

    With ``cfg.SIDE_PANEL`` off, the overlaid clip alone at its own size.
    """
    if cfg.SIDE_PANEL:
        panel = Panel(signal, analysis, info.width, info.height, cfg=cfg)
        out_w = info.width * 2
        p_at = panel.p_at
    else:
        # No panel to agree with, but the read box keeps the color it would show:
        # the same display smoothing of the same signal.
        panel = None
        out_w = info.width
        p_display = display_smooth(signal.per_frame, signal.fps, cfg.PANEL_P_SMOOTH_SECONDS)

        def p_at(frame: int) -> float:
            return float(p_display[min(frame, len(p_display) - 1)])

    edges, points = visible_parts(cfg.DRAW_FACE)
    if cfg.DRAW_JOINTS:
        # Only the chain the story is about: an edge is drawn when both of its
        # ends are kept joints, so shoulder-hip-knee keeps both sides' torso and
        # thigh and drops the arms, the shins and the cross-body lines.
        keep = {j for j, name in enumerate(KPT_NAMES)
                if name.split("_", 1)[-1] in cfg.DRAW_JOINTS}
        edges = [(a, b, side) for a, b, side in edges
                 if a in keep and b in keep and side in ("L", "R")]
        points = frozenset(points & keep)
    # Far side first, near side last: the two nearly overlap side-on, and the
    # side drawn last reads as the one in front. The near side is the one the
    # hinge is measured on (the side ViTPose is surer of), so it goes on top.
    near = "L" if analysis.hinge_side == "left" else "R"
    edges = sorted(edges, key=lambda e: e[2] == near)
    scale = info.width / 720
    thickness = max(1, round(cfg.LINE_THICKNESS * scale))
    radius = max(1, round(cfg.POINT_RADIUS * scale))
    stroke = max(1, round(2 * scale))
    font = resolve_font(cfg.PANEL_FONT, cfg.PANEL_FONT_INDEX)
    comet = Comet(width=cfg.BAR_TRAIL_WIDTH * scale, taper=cfg.BAR_TRAIL_TAPER,
                  opacity=cfg.BAR_TRAIL_OPACITY, fade=cfg.BAR_TRAIL_FADE,
                  glow=cfg.BAR_TRAIL_GLOW, glow_opacity=cfg.BAR_TRAIL_GLOW_OPACITY,
                  softness=cfg.BAR_TRAIL_SOFTNESS) if cfg.BAR_PATH_SECONDS else None

    # The crop System One read at each frame is the nearest sampled one.
    crop_keys = np.array(sorted(crops)) if crops else np.zeros(0, int)

    cap = cv2.VideoCapture(str(info.path))
    writer = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), info.fps,
                             (out_w, info.height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"OpenCV could not open a writer for {dst}")

    # Pose, plate and the back signal are indexed by the inference clip. An export
    # with a different frame count is matched by index only as far as both go.
    limit = min(info.n_frames, len(signal.per_frame), len(analysis.hip_deg))
    if plate is not None:
        limit = min(limit, len(plate.per_frame_cx))

    written = posed = 0
    try:
        with Progress(*_PROGRESS_COLUMNS, console=console, transient=True) as progress:
            task = progress.add_task("render", total=limit)
            for i in range(limit):
                ok, frame = cap.read()
                if not ok:
                    break
                h, w = frame.shape[:2]

                if len(crop_keys):
                    x0, y0, x1, y1 = crops[int(crop_keys[np.argmin(np.abs(crop_keys - i))])]
                    p = p_at(i)   # the box agrees with the panel beside it
                    color = (107, 107, 255) if p > cfg.THRESHOLD else (102, 207, 81)
                    cv2.rectangle(frame, (int(x0 * w), int(y0 * h)), (int(x1 * w), int(y1 * h)),
                                  color, stroke, cv2.LINE_AA)

                _draw_plate(frame, plate, i, cfg=cfg, stroke=stroke, fps=info.fps, comet=comet)

                person = _person_at(by_index, i, hold=2)
                if person is not None:
                    draw_person(frame, person, w, h, edges=edges, points=points,
                                thickness=thickness, radius=radius, draw_bbox=False)
                    if cfg.DRAW_HIP_ANGLE:
                        _draw_hinge(frame, person, float(analysis.hip_deg[i]),
                                    side=analysis.hinge_side, facing=analysis.facing,
                                    clockwise=analysis.arc_clockwise,
                                    cfg=cfg, scale=scale, font=font)
                    posed += 1

                writer.write(frame if panel is None else np.hstack([frame, panel.render(i)]))
                written += 1
                progress.update(task, advance=1)
    finally:
        cap.release()
        writer.release()
    return {"frames_written": written, "frames_with_pose": posed,
            "output": f"{out_w}x{info.height}"}


def _mark_reps(ax, t: np.ndarray, y: np.ndarray, reps, *, cfg) -> None:
    """The timed span on every rep, drawn the way chin_ups' displacement plot does.

    Shaded from the start of the pull to lockout, the interval `lift` reports,
    with a bracket and its length above, both ends stamped so it can be checked
    by hand, and the verdict under the bracket. The hold and the lowering are
    shaded grey: part of the rep, not part of the number.
    """
    finite = y[np.isfinite(y)]
    top = float(finite.max()) if len(finite) else 1.0
    base = float(finite.min()) if len(finite) else 0.0
    rng = max(top - base, 1e-6)
    bar = top + rng * 0.16          # one height for every bracket, so they read as a row
    ax.set_ylim(base - rng * 0.22, bar + rng * 0.26)

    def at(time: float) -> float:
        return float(y[int(np.argmin(np.abs(t - time)))])

    for i, rep in enumerate(reps):
        a, b = rep.onset_time, rep.peak_time
        verdict = "#e03131" if rep.back_label == cfg.POSITIVE_LABEL else "#2f9e44"
        ax.axvspan(a, b, color="#2f6df6", alpha=0.13, zorder=0,
                   label="pull, timed (start to lockout)" if i == 0 else None)
        ax.axvspan(b, rep.end_time, color="#8a8f98", alpha=0.08, zorder=0,
                   label="hold + lower (not timed)" if i == 0 else None)
        for x in (a, b):
            ax.axvline(x, color="#2f6df6", ls=":", lw=1.1, alpha=0.65, zorder=2)

        ax.annotate("", xy=(b, bar), xytext=(a, bar),
                    arrowprops=dict(arrowstyle="<|-|>", color="#1864ab", lw=1.6,
                                    shrinkA=0, shrinkB=0))
        ax.text((a + b) / 2, bar + rng * 0.025, f"{rep.lift_seconds:.2f}s", ha="center",
                va="bottom", fontsize=11, fontweight="bold", color="#1864ab")
        if rep.back_label:
            ax.text((a + b) / 2, bar - rng * 0.03, f"{rep.back_label} {rep.back_p:.0%}",
                    ha="center", va="top", fontsize=8.5, fontweight="bold", color=verdict)

        a_y, b_y = at(a), at(b)
        ax.plot(b, b_y, "o", color="#e8590c", ms=8, zorder=4)
        ax.plot(a, a_y, "o", color="#8a8f98", ms=6, zorder=4)
        ax.annotate(f"{rep.number}", (b, b_y), textcoords="offset points", xytext=(-11, 3),
                    ha="right", fontsize=11, fontweight="bold", color="#e8590c")
        ax.annotate(f"{b:.2f}s", (b, b_y), textcoords="offset points", xytext=(8, -14),
                    ha="left", fontsize=9, color="#e8590c")
        ax.annotate(f"{a:.2f}s", (a, a_y), textcoords="offset points", xytext=(0, -16),
                    ha="center", fontsize=9, color="#5c6470")


def plot_timeline(signal, reads, analysis, dst: Path, *, cfg) -> None:
    """Bar height, hip angle and P(rounded), with every rep marked on each.

    The signal the reps were counted on carries the timed span of each rep;
    the others are shaded start to back down in the rep's verdict. The two rep
    signals share a time axis so they can be checked against each other: solid
    lines are the lockouts of the reps in use, dotted ones the other signal's.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.arange(len(signal.per_frame)) / signal.fps
    fig, (ax, axh, ax2) = plt.subplots(3, 1, figsize=(13, 11), sharex=True,
                                       gridspec_kw={"height_ratios": [1.5, 1, 1]})
    timed = ax if analysis.source == "plate" else axh

    for rep in analysis.reps:
        color = "#ff6b6b" if rep.back_label == cfg.POSITIVE_LABEL else "#51cf66"
        for a in (ax, axh, ax2):
            if a is not timed:
                a.axvspan(rep.onset_time, rep.end_time, color=color, alpha=0.13, zorder=0)
        ax2.axvspan(rep.onset_time, rep.knee_frame / signal.fps, color=color, alpha=0.5,
                    zorder=0, hatch="//", fill=False, lw=0)

    # Bar height, or a note that there was none.
    if analysis.has_plate:
        ax.plot(t, analysis.raw_cm / 100, color="#c9ced6", lw=1, label="interpolated")
        ax.plot(t, analysis.height_cm / 100, color="#2f6df6", lw=2.2, label="smoothed")
        ax.axhline(0, color="#8a8f98", ls="--", lw=1.2, label="floor")
        if analysis.source == "plate":
            _mark_reps(ax, t, analysis.height_cm / 100, analysis.reps, cfg=cfg)
    else:
        ax.text(0.5, 0.5, "no plate track: reps counted off the hip hinge", ha="center",
                va="center", transform=ax.transAxes, color="#868e96")
    ax.set_ylabel(f"bar height (m)\nassumes a {cfg.PLATE_DIAMETER_CM / 100:g} m plate")
    lifts = [r.lift_seconds for r in analysis.reps]
    mean = f", {np.mean(lifts):.2f}s mean pull" if lifts else ""
    ax.set_title(f"{analysis.count} reps off the {analysis.source}{mean} · "
                 f"{cfg.PLATE_MODEL} plate, {cfg.POSE_MODEL.split('/')[-1]} pose",
                 fontsize=13, fontweight="bold", pad=28)
    ax.grid(alpha=0.25)
    # Above the axes, not inside them: any in-plot corner eventually sits on a rep.
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=5, fontsize=9, frameon=False)

    # Hip angle, with both signals' lockouts.
    axh.plot(t, analysis.hip_deg, color="#7048e8", lw=2, label="hip angle")
    if analysis.source == "hinge":
        _mark_reps(axh, t, analysis.hip_deg, analysis.reps, cfg=cfg)
    for reps_, name in ((analysis.plate_reps, "plate"), (analysis.hinge_reps, "hinge")):
        style = "-" if name == analysis.source else ":"
        for k, rep in enumerate(reps_):
            axh.axvline(rep.peak_time, color="#2f6df6" if name == "plate" else "#7048e8",
                        ls=style, lw=1.2, alpha=0.8,
                        label=f"{name} lockout" if k == 0 else None)
    axh.set_ylabel("hip angle (°)\nshoulder-hip-knee")
    if analysis.source != "hinge":
        axh.set_ylim(0, 190)
    axh.grid(alpha=0.25)
    axh.legend(loc="lower right", fontsize=9, frameon=False)

    ax2.scatter(signal.read_times, signal.raw, s=10, color="#adb5bd", zorder=2,
                label=f"reads ({len(signal.raw)})")
    ax2.plot(t, signal.per_frame, color="#a5c0f7", lw=1.2, zorder=3,
             label=f"median {cfg.SMOOTH_READS}")
    ax2.plot(t, display_smooth(signal.per_frame, signal.fps, cfg.BACK_SMOOTH_SECONDS),
             color="#2f6df6", lw=2, zorder=3,
             label=f"judged (σ {cfg.BACK_SMOOTH_SECONDS:g}s)")
    ax2.axhline(cfg.THRESHOLD, color="#8a8f98", ls="--", lw=1.2)
    failed = [r.time for r in reads if not r.ok]
    if failed:
        ax2.scatter(failed, [0.02] * len(failed), marker="x", color="#c92a2a", zorder=4,
                    label=f"failed ({len(failed)})")
    ax2.set_ylim(-0.02, 1.02)
    ax2.set_xlim(0, t[-1] if len(t) else 1)
    ax2.set_xlabel("time (s)")
    ax2.set_ylabel(f"P({cfg.POSITIVE_LABEL})")
    ax2.set_title(f"{cfg.READ_MODEL} · System One · hatched: the window each verdict averages",
                  fontsize=10)
    ax2.grid(alpha=0.25)
    ax2.legend(loc="upper right", fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(dst, dpi=150)
    plt.close(fig)
