"""The export video, the timeline plot and the elbow arcs."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn

from .panel import Panel
from .skeleton import KPT_NAMES, draw_person, visible_parts
from .video import VideoInfo

_INDEX = {name: i for i, name in enumerate(KPT_NAMES)}

_PROGRESS_COLUMNS = (
    TextColumn("[cyan]rendering[/]"),
    BarColumn(),
    TaskProgressColumn(),
    TextColumn("{task.completed}/{task.total} frames"),
    TimeElapsedColumn(),
)


def _hex(bgr) -> str:
    return "#{:02x}{:02x}{:02x}".format(*tuple(bgr)[::-1])


def _draw_elbows(frame, person, *, arms, radius: int, stroke: int, color) -> None:
    """Each elbow angle as a filled arc between upper arm and forearm.

    One neutral color, so the arcs read as a measurement, not more skeleton.
    """
    h, w = frame.shape[:2]
    kpts = person.get("kpts_xy") or []
    layer = None
    outlines = []
    for shoulder, elbow, wrist in arms:
        idx = [_INDEX[n] for n in (shoulder, elbow, wrist)]
        if max(idx) >= len(kpts) or any(tuple(kpts[i]) == (0, 0) for i in idx):
            continue
        s, e, wr = (np.array(kpts[i], float) * [w, h] for i in idx)
        a_s = float(np.degrees(np.arctan2(*(s - e)[::-1])))
        a_w = float(np.degrees(np.arctan2(*(wr - e)[::-1])))
        # The interior angle: a sweep within +/-180 of the start is the short arc.
        sweep = (a_w - a_s + 180) % 360 - 180
        c = (int(e[0]), int(e[1]))
        if layer is None:
            layer = frame.copy()
        cv2.ellipse(layer, c, (radius, radius), 0, a_s, a_s + sweep, color, -1, cv2.LINE_AA)
        outlines.append((c, a_s, a_s + sweep))
    if layer is None:
        return
    cv2.addWeighted(layer, 0.35, frame, 0.65, 0, dst=frame)
    for c, a, b in outlines:
        cv2.ellipse(frame, c, (radius, radius), 0, a, b, color, stroke, cv2.LINE_AA)


def _arc_colors(analysis, n_frames: int, fps: float, cfg) -> dict:
    """Frame -> elbow arc color, for ``cfg.ELBOW_ARC_GRADE_COLORS``.

    Around each graded rep's valley, through the dead hang to its onset, the arc
    takes that rep's bottom grade; elsewhere ``cfg.ELBOW_ARC_COLOR`` applies.
    """
    if not cfg.ELBOW_ARC_GRADE_COLORS:
        return {}
    hold = max(1, round(0.25 * fps))   # long enough to read even without a hang
    colors = {}
    for rep in analysis.reps:
        if rep.rom is None:
            continue
        color = cfg.PANEL_FULL if rep.rom.bottom_ok else cfg.PANEL_PARTIAL
        last = min(n_frames - 1, max(rep.onset_frame, rep.valley_frame + hold))
        for i in range(max(0, rep.valley_frame - hold), last + 1):
            colors[i] = color
    return colors


def _person_at(by_index: dict, i: int, hold: int):
    """This frame's athlete, or the last one seen within *hold* frames."""
    for j in range(i, max(-1, i - hold - 1), -1):
        persons = by_index.get(j)
        if persons:
            return persons[0]
    return None


def render(info: VideoInfo, *, by_index, analysis, n_frames: int, fps: float, dst: Path,
           cfg, console) -> dict:
    """Write the overlaid clip beside its panel.

    *n_frames* and *fps* are the inference clip's; frames are matched by index.
    """
    # The panel is laid out for portrait, 9:16 at most. A landscape clip's width
    # would make its width-scaled text and markers too tall for the frame.
    panel_w = min(info.width, round(info.height * 9 / 16 / 2) * 2)
    panel = Panel(analysis, n_frames, fps, panel_w, info.height, cfg=cfg)
    out_w = info.width + panel_w

    edges, points = visible_parts(cfg.DRAW_FACE)
    scale = info.width / 720
    thickness = max(1, round(cfg.LINE_THICKNESS * scale))
    radius = max(1, round(cfg.POINT_RADIUS * scale))
    arc = max(4, round(cfg.ELBOW_ARC_RADIUS * scale))
    stroke = max(1, round(2 * scale))
    arc_colors = _arc_colors(analysis, n_frames, fps, cfg)

    cap = cv2.VideoCapture(str(info.path))
    writer = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), info.fps,
                             (out_w, info.height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"OpenCV could not open a writer for {dst}")

    limit = min(info.n_frames, n_frames)
    written = posed = 0
    try:
        with Progress(*_PROGRESS_COLUMNS, console=console, transient=True) as progress:
            task = progress.add_task("render", total=limit)
            for i in range(limit):
                ok, frame = cap.read()
                if not ok:
                    break
                h, w = frame.shape[:2]

                person = _person_at(by_index, i, hold=2)
                if person is not None:
                    draw_person(frame, person, w, h, edges=edges, points=points,
                                thickness=thickness, radius=radius,
                                draw_bbox=cfg.DRAW_BBOX, bbox_color=cfg.BBOX_COLOR,
                                colors=cfg.SKELETON_COLORS)
                    if cfg.DRAW_ELBOW_ANGLES:
                        _draw_elbows(frame, person, arms=cfg.ARM_JOINTS, radius=arc,
                                     stroke=stroke,
                                     color=arc_colors.get(i, cfg.ELBOW_ARC_COLOR))
                    posed += 1

                writer.write(np.hstack([frame, panel.render(i)]))
                written += 1
                progress.update(task, advance=1)
    finally:
        cap.release()
        writer.release()
    return {"frames_written": written, "frames_with_pose": posed,
            "output": f"{out_w}x{info.height}"}


def _figure_rc(figure) -> dict:
    """matplotlib rc for ``cfg.TIMELINE_FIGURE``; empty (matplotlib's own look) for None."""
    if not figure:
        return {}
    bg, text, grid = (_hex(figure[k]) for k in ("background", "text", "grid"))
    return {"figure.facecolor": bg, "axes.facecolor": bg, "savefig.facecolor": bg,
            "text.color": text, "axes.labelcolor": text, "xtick.color": text,
            "ytick.color": text, "axes.edgecolor": grid, "grid.color": grid}


def plot_timeline(analysis, dst: Path, *, cfg, title: str = "") -> None:
    """Displacement with each rep's measured interval, plus the two ROM traces and cutoffs.

    The span follows ``cfg.REP_METRIC`` so the graph and the video quote the same number.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with plt.rc_context(_figure_rc(cfg.TIMELINE_FIGURE)):
        _plot_timeline(analysis, dst, cfg=cfg, title=title)


def _plot_timeline(analysis, dst: Path, *, cfg, title: str) -> None:
    import matplotlib.pyplot as plt

    from .reps import phase_label, rep_value

    metric = cfg.REP_METRIC
    # Verdict colors, shared by every axis so a grade's color means the same everywhere.
    c = {role: _hex(bgr) for role, bgr in cfg.TIMELINE_COLORS.items()}
    good, bad = c["full"], c["partial"]
    # A dark figure's grid is its own color; a light one's is matplotlib's, faded.
    grid_alpha = 1.0 if cfg.TIMELINE_FIGURE else 0.25

    def span(rep):
        """(start, end) of the interval this rep's number actually measures."""
        if metric == "moving":
            return rep.onset_time, rep.peak_time
        if metric == "total":
            return rep.valley_time, rep.end_time
        return rep.valley_time, rep.peak_time

    t = analysis.times
    fig, (ax, ax2, ax3) = plt.subplots(
        3, 1, figsize=(13, 9), sharex=True,
        gridspec_kw={"height_ratios": [3, 1, 1]},
    )

    def at(time: float) -> int:
        """Array position of the sample nearest *time*."""
        return int(np.argmin(np.abs(t - time)))

    def verdict_color(rep) -> str:
        return good if rep.complete else bad

    ax.plot(t, analysis.raw - analysis.baseline, color=c["raw"], lw=1,
            label="raw (torso mean)", zorder=1)
    ax.plot(t, analysis.displacement, color=c["line"], lw=2.2, label="smoothed", zorder=3)
    ax.axhline(0, color="#8a8f98", ls="--", lw=1.2, zorder=2,
               label="relative zero (hanging)")

    top = float(np.max(analysis.displacement)) if len(analysis.displacement) else 1.0
    bar = top * 1.16          # one height for every bracket, so they read as a row
    ax.set_ylim(bottom=-top * 0.22, top=bar * 1.14)   # room for the valley timestamps

    for i, rep in enumerate(analysis.reps):
        a, b = span(rep)
        value = rep_value(rep, metric)

        ax.axvspan(a, b, color=c["line"], alpha=0.13, zorder=0,
                   label=phase_label(metric) if i == 0 else None)
        for x in (a, b):
            ax.axvline(x, color=c["line"], ls=":", lw=1.1, alpha=0.65, zorder=2)

        # The measured interval as a bracket.
        ax.annotate("", xy=(b, bar), xytext=(a, bar),
                    arrowprops=dict(arrowstyle="<|-|>", color=c["interval"], lw=1.6,
                                    shrinkA=0, shrinkB=0))
        ax.text((a + b) / 2, bar + top * 0.025, f"{value:.2f}s", ha="center",
                va="bottom", fontsize=11, fontweight="bold", color=c["interval"])

        # When timing starts at onset, show the excluded dead hang before it.
        if metric == "moving" and a - rep.valley_time > 1.5 / 30:
            ax.axvspan(rep.valley_time, a, color="#8a8f98", alpha=0.10, zorder=0,
                       label="dead hang (excluded)" if i == 0 else None)
            ax.plot(rep.valley_time, analysis.displacement[at(rep.valley_time)],
                    "o", color="#adb5bd", ms=5, zorder=4)
            ax.annotate(f"hang {a - rep.valley_time:.2f}s",
                        ((rep.valley_time + a) / 2, 0), textcoords="offset points",
                        xytext=(0, -30), ha="center", fontsize=8.5, color="#868e96")

        a_y = analysis.displacement[at(a)]
        ax.plot(b, rep.height, "o", color=c["peak"], ms=8, zorder=4)
        ax.plot(a, a_y, "o", color="#8a8f98", ms=6, zorder=4)

        # The rep number carries its verdict mark.
        mark = "" if rep.rom is None else ("  \u2713" if rep.complete else "  \u2717")
        ax.annotate(f"{rep.number}{mark}", (b, rep.height), textcoords="offset points",
                    xytext=(11, 3), ha="left", fontsize=11, fontweight="bold",
                    color=verdict_color(rep))

        # Both endpoints stamped, so the bracket above is checkable by hand.
        ax.annotate(f"{b:.2f}s", (b, rep.height), textcoords="offset points",
                    xytext=(0, 13), ha="center", fontsize=9.5, color=c["peak"])
        ax.annotate(f"{a:.2f}s", (a, a_y), textcoords="offset points",
                    xytext=(0, -17), ha="center", fontsize=9.5, color="#5c6470")

    ax.set_ylabel("vertical displacement\n(fraction of frame height)")
    ax.grid(alpha=grid_alpha)
    # Above the axes, not inside them: any in-plot corner eventually sits on a rep.
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.015), ncol=5,
              fontsize=9, frameon=False)
    ax.set_title(title or f"{analysis.count} reps", fontsize=13, fontweight="bold",
                 pad=28)

    # ROM traces are masked to frames with hands on the bar; after letting go the
    # values are meaningless and would dominate the axes.
    on_bar = analysis.on_bar if len(analysis.on_bar) == len(t) else np.ones(len(t), bool)

    def held(series):
        return np.where(on_bar, series, np.nan)

    def threshold(axis, y, text, *, color=c["cutoff"], style="--", width=1.2):
        """A cutoff line, labelled in the legend row above the axis.

        In-plot labels would overlap the data or each other; above the axis is always clear.
        """
        axis.axhline(y, color=color, ls=style, lw=width, zorder=2, label=text)

    def label_row(axis) -> None:
        axis.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=5,
                    fontsize=8.5, frameon=False, handlelength=2.4,
                    borderpad=0, borderaxespad=0.15)

    # Bottom of the rep: elbow angle, graded at each valley. Each arm faintly in
    # its skeleton color; the grade reads their mean.
    arm_colors = (_hex(cfg.SKELETON_COLORS["left_arm"]), _hex(cfg.SKELETON_COLORS["right_arm"]))
    for arm, color, name in zip(analysis.arms, arm_colors, ("Left", "Right")):
        ax2.plot(t, held(arm), color=color, lw=1.0, alpha=0.7, label=f"{name.lower()} arm")
    ax2.plot(t, held(analysis.elbow), color=c["elbow"], lw=1.6, label="mean (graded)")
    # The cutoff reps were held to (profile if calibrated, else this clip's). The
    # absolute floor applies only when uncalibrated, so it is only drawn then.
    required = analysis.hang_required
    source = "profile" if analysis.calibrated else "this clip"
    if np.isfinite(required):
        threshold(ax2, required, f"full hang, {source} ({required:.0f}\u00b0)")
        if not analysis.calibrated and required > cfg.ROM_HANG_ELBOW_DEGREES + 0.5:
            threshold(ax2, cfg.ROM_HANG_ELBOW_DEGREES,
                      f"floor ({cfg.ROM_HANG_ELBOW_DEGREES:.0f}\u00b0)",
                      color="#868e96", style=":", width=1.0)
    for rep in analysis.reps:
        i = at(rep.valley_time)
        if np.isfinite(analysis.elbow[i]):
            ax2.plot(rep.valley_time, analysis.elbow[i], "o", ms=7, zorder=4,
                     color=good if rep.rom is None or rep.rom.bottom_ok else bad)
    ax2.set_ylabel("elbow angle\n(degrees)")
    ax2.invert_yaxis()   # as on the panel: the line rises as the body does
    label_row(ax2)
    ax2.grid(alpha=grid_alpha)

    # Top of the rep: head clearance, in torso lengths so camera distance cancels out.
    clearance = held(analysis.clearance)
    needs = (analysis.clearance_required if np.isfinite(analysis.clearance_required)
             else cfg.ROM_CLEARANCE_TORSO_FRACTION)
    ax3.plot(t, clearance, color=c["clearance"], lw=1.6)
    threshold(ax3, needs, f"clears the bar, {source} ({needs:.2f})")
    ax3.axhline(0, color="#adb5bd", ls=":", lw=1.0, zorder=1)
    ax3.fill_between(t, needs, clearance, where=clearance > needs,
                     color=c["clearance"], alpha=0.18)
    for rep in analysis.reps:
        i = at(rep.peak_time)
        if np.isfinite(analysis.clearance[i]):
            ax3.plot(rep.peak_time, analysis.clearance[i], "o", ms=7, zorder=4,
                     color=good if rep.rom is None or rep.rom.top_ok else bad)
    ax3.set_ylabel("head above hands\n(torso lengths)")
    ax3.set_xlabel("time (s)")
    label_row(ax3)
    ax3.grid(alpha=grid_alpha)

    fig.subplots_adjust(left=0.09, right=0.98, top=0.89, bottom=0.07, hspace=0.30)
    fig.savefig(dst, dpi=150)
    plt.close(fig)
