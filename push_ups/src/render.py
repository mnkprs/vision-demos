"""Drawing the returned poses back onto the frames."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn

from .panel import SidePanel
from .skeleton import KPT_NAMES, draw_hud, draw_person, visible_parts
from .video import VideoInfo

_PROGRESS_COLUMNS = (
    TextColumn("[cyan]rendering[/]"),
    BarColumn(),
    TaskProgressColumn(),
    TextColumn("{task.completed}/{task.total} frames"),
    TimeElapsedColumn(),
)


def panel_width(info: VideoInfo) -> int:
    """The panel's width: the clip's own for portrait, at most 9:16 of its height.

    The panel's text scales with its width. A landscape clip's full width would
    make that text too tall for the frame, so the panel stays portrait-shaped,
    as in pull_ups.
    """
    return min(info.width, round(info.height * 9 / 16 / 2) * 2)


def render(
    info: VideoInfo,
    by_index: dict[int, list[dict]],
    n_returned: int,
    dst: Path,
    *,
    cfg,
    console,
    analysis=None,
) -> dict:
    """Write an overlay video and return per-run stats.

    When the clip was decimated (more source frames than the model posed), the
    frames in between have no result of their own. Holding the last pose across
    the gap beats flickering, and those frames are labelled `(held)` so a
    measured frame is never confused with a carried one.
    """
    edges, points = visible_parts(cfg.DRAW_FACE)

    # Stroke widths are pixels, so a fixed value thins out as the export grows.
    # Scaling against the same 720px reference the panel uses keeps the overlay
    # looking identical at any export size.
    stroke = info.width / 720
    thickness = max(1, round(cfg.LINE_THICKNESS * stroke))
    radius = max(1, round(cfg.POINT_RADIUS * stroke))

    stride = max(1, round(info.n_frames / max(n_returned, 1)))
    hold_limit = stride - 1

    panel = None
    out_w = info.width
    if cfg.SIDE_PANEL and analysis is not None:
        # Same height as the clip, which keeps its native aspect ratio.
        panel_w = panel_width(info)
        panel = SidePanel(analysis, info.n_frames, info.fps, panel_w, info.height, cfg=cfg)
        out_w = info.width + panel_w
        console.print(f"  side panel: {panel.states} states"
                      + (" + live block" if panel.live is not None else "")
                      + f", output {out_w}x{info.height}")

    cap = cv2.VideoCapture(str(info.path))
    writer = cv2.VideoWriter(
        str(dst), cv2.VideoWriter_fourcc(*"mp4v"), info.fps, (out_w, info.height)
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"OpenCV could not open a writer for {dst}")

    last_persons: list[dict] | None = None
    held_for = 0
    n_drawn = n_held = n_people = 0

    try:
        with Progress(*_PROGRESS_COLUMNS, console=console, transient=True) as progress:
            task = progress.add_task("render", total=info.n_frames)
            for i in range(info.n_frames):
                ok, frame = cap.read()
                if not ok:
                    break

                if i in by_index:
                    persons, held = by_index[i], False
                    last_persons, held_for = persons, 0
                elif last_persons is not None and held_for < hold_limit:
                    persons, held = last_persons, True
                    held_for += 1
                else:
                    persons, held = [], False

                h, w = frame.shape[:2]
                for person in persons:
                    draw_person(
                        frame, person, w, h,
                        edges=edges, points=points,
                        thickness=thickness, radius=radius,
                        held=held, draw_bbox=cfg.DRAW_BBOX, draw_label=cfg.DRAW_TRACK_LABEL,
                        bbox_color=cfg.BBOX_COLOR, colors=cfg.SKELETON_COLORS,
                    )

                if persons:
                    n_drawn += 1
                    n_people += len(persons)
                    n_held += bool(held)

                if cfg.DRAW_HUD:
                    hud = f"frame {i}/{info.n_frames}  |  {len(persons)} person(s)"
                    draw_hud(frame, hud + ("  (held)" if held else ""))

                if panel is not None:
                    frame = np.hstack([frame, panel.draw(i)])

                writer.write(frame)
                progress.update(task, advance=1)
    finally:
        cap.release()
        writer.release()

    return {
        "frames_written": info.n_frames,
        "frames_with_pose": n_drawn,
        "frames_held": n_held,
        "mean_people_per_posed_frame": round(n_people / n_drawn, 2) if n_drawn else 0.0,
        "hold_limit": hold_limit,
    }


def _shade_off_plank(ax, t, in_plank) -> None:
    """Grey out the frames the plank gate excluded, so a missing rep has a visible reason."""
    i = 0
    first = True
    while i < len(in_plank):
        if in_plank[i]:
            i += 1
            continue
        j = i
        while j < len(in_plank) and not in_plank[j]:
            j += 1
        ax.axvspan(t[i], t[min(j, len(t) - 1)], color="#8a8f98", alpha=0.18, zorder=0,
                   label="not in a plank (ignored)" if first else None)
        first = False
        i = j


def plot_displacement(analysis, dst: Path, *, cfg, title: str = "") -> None:
    """Depth, elbow angle and body line over time, every rep marked.

    The top row is what counts the reps: each one is shaded from where the
    descent starts to where the press locks out, the span its times measure.
    The two rows under it are what grade them, with the cutoffs drawn in, so
    every full, partial, sagging or piked verdict can be checked by eye.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = analysis.times
    fig, (ax, ax2, ax3) = plt.subplots(
        3, 1, figsize=(13, 9), sharex=True,
        gridspec_kw={"height_ratios": [3, 1.4, 1.4]},
    )

    ax.plot(t, analysis.top - analysis.raw, color="#c9ced6", lw=1,
            label="raw (shoulders vs wrists)", zorder=1)
    ax.plot(t, analysis.depth, color="#2f6df6", lw=2.2, label="smoothed", zorder=3)
    ax.axhline(0, color="#8a8f98", ls="--", lw=1.2, zorder=2, label="relative zero (top)")
    for a in (ax, ax2, ax3):
        _shade_off_plank(a, t, analysis.in_plank)

    seen = analysis.depth[np.isfinite(analysis.depth)]
    deepest = float(np.max(seen)) if len(seen) else 1.0
    bar = deepest * 1.16          # one height for every bracket, so they read as a row
    ax.set_ylim(bottom=-deepest * 0.22, top=bar * 1.14)
    ax.invert_yaxis()             # down the screen is down the push-up

    for i, rep in enumerate(analysis.reps):
        a, b = rep.onset_time, rep.lockout_time
        color = "#2b8a3e" if rep.full_depth else "#c92a2a"
        ax.axvspan(a, b, color=color, alpha=0.10, zorder=0,
                   label="rep, onset to lockout (green full, red partial)" if i == 0 else None)
        ax.axvline(rep.bottom_time, color=color, ls=":", lw=1.1, alpha=0.65, zorder=2)

        # The measured interval, drawn as an interval rather than implied by a
        # number floating near a bottom.
        ax.annotate("", xy=(b, bar), xytext=(a, bar),
                    arrowprops=dict(arrowstyle="<|-|>", color="#1864ab", lw=1.6,
                                    shrinkA=0, shrinkB=0))
        ax.text((a + b) / 2, bar + deepest * 0.025, f"{rep.down_seconds:.2f} + "
                f"{rep.up_seconds:.2f}s", ha="center", va="top", fontsize=10,
                fontweight="bold", color="#1864ab")

        ax.plot(rep.bottom_time, rep.depth, "o", color=color, ms=8, zorder=4)
        ax.annotate(f"{rep.number}", (rep.bottom_time, rep.depth), textcoords="offset points",
                    xytext=(11, -3), ha="left", fontsize=11, fontweight="bold", color=color)
        ax.annotate(f"{rep.bottom_time:.2f}s", (rep.bottom_time, rep.depth),
                    textcoords="offset points", xytext=(0, -14), ha="center", fontsize=9.5,
                    color=color)

    ax.set_ylabel("shoulder depth below the top\n(fraction of frame height)")
    ax.grid(alpha=0.25)
    # Above the axes, not inside them: any in-plot corner eventually sits on a rep.
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.015), ncol=5, fontsize=9, frameon=False)
    ax.set_title(title or f"{analysis.count} reps", fontsize=13, fontweight="bold", pad=28)

    full = analysis.full_depth_degrees
    ax2.plot(t, analysis.elbow, color="#5c6470", lw=1.4)
    ax2.axhline(full, color="#2b8a3e", ls="--", lw=1.2)
    ax2.fill_between(t, full, analysis.elbow, where=analysis.elbow <= full,
                     color="#2b8a3e", alpha=0.18)
    for rep in analysis.reps:
        if np.isfinite(rep.elbow_degrees):
            ax2.plot(rep.bottom_time, rep.elbow_degrees, "o", ms=6,
                     color="#2b8a3e" if rep.full_depth else "#c92a2a")
    ax2.set_ylabel(f"elbow angle (°)\n{analysis.elbow_side} arm")
    ax2.grid(alpha=0.25)

    tol = analysis.body_line_tolerance
    ax3.plot(t, analysis.hip_line, color="#5c6470", lw=1.4)
    ax3.axhspan(-tol, tol, color="#2b8a3e", alpha=0.10)
    ax3.axhline(0, color="#8a8f98", ls="--", lw=1)
    for rep in analysis.reps:
        # Each rep's verdict, dotted where it was read: the furthest the hips
        # stayed off the line, anywhere in the rep or the rest after it.
        if np.isfinite(rep.hip_degrees):
            ax3.plot(rep.hip_time, rep.hip_degrees, "o", ms=6,
                     color="#2b8a3e" if rep.body_line == "straight" else "#c92a2a")
    ax3.set_ylabel(f"hips off line (°)\n+ sag, − hips high, to {analysis.hip_reference}")
    ax3.set_xlabel("time (s)")
    ax3.grid(alpha=0.25)

    fig.subplots_adjust(left=0.09, right=0.98, top=0.89, bottom=0.07, hspace=0.12)
    fig.savefig(dst, dpi=150)
    plt.close(fig)


def plot_joint(frames: list[dict], fps: float, joint: str, dst: Path) -> bool:
    """Plot one joint's height per track. Returns False if it never appeared.

    `track_id` is what makes this meaningful: without it `persons[0]` is a list
    slot, not a person, and the series would jump between people.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    j = KPT_NAMES.index(joint)

    tracks: dict = {}
    for f in frames:
        for p in f["persons"]:
            kpts = p.get("kpts_xy", [])
            if j < len(kpts) and tuple(kpts[j]) != (0, 0):
                # y grows downward in image coordinates; flip so "up" reads as up.
                tracks.setdefault(p.get("track_id"), []).append(
                    (f["index"] / fps, 1.0 - kpts[j][1])
                )

    if not tracks:
        return False

    fig, ax = plt.subplots(figsize=(11, 3.5))
    for tid, series in sorted(tracks.items(), key=lambda kv: (kv[0] is None, kv[0])):
        ts, ys = zip(*series)
        ax.plot(ts, ys, lw=1.5, label=f"track {tid}")

    ax.set_xlabel("time (s)")
    ax.set_ylabel(f"{joint} height (normalized, up = higher)")
    ax.set_title(f"{joint} over time")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(dst, dpi=140)
    plt.close(fig)
    return True
