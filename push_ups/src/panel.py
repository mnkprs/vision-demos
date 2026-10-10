"""The side panel: rep count, rep times, the depth tally, a per-rep plot, and a live block.

Two layers, because they change at different rates. Everything per rep is known
before the first frame is written, so that layer is rendered once per distinct
count (seven images for a six-rep clip) with matplotlib, rather than redrawn on
every frame. That also lets the axes be fixed up front, since the x axis is the
rep *number* and the y limits span every rep in the clip. Nothing rescales as
dots appear, which is what makes the reveal read as a graph filling in rather
than a graph thrashing.

The live block moves every frame: the elbow angle, the phase of the rep, the rep
timer and the depth. A matplotlib figure per frame would cost more than the
whole render, so, as in running/, it is drawn with OpenCV and Pillow into its
own band of a copy of the per-rep image, over a background drawn once.

Top to bottom: the headline numbers, the live block, the graph, the closing note.
"""

from __future__ import annotations

import warnings
from bisect import insort
from dataclasses import dataclass, fields
from functools import lru_cache

import cv2
import numpy as np

from .reps import (BODY_LINE_LABELS, EXTREME_WORDS, METRICS, body_line_note, extremes,
                   format_value, rep_value)
from .text import resolve_font

def hex_from_bgr(bgr) -> str:
    """``(b, g, r)``, as config holds colors, -> ``"#rrggbb"``."""
    return "#{:02x}{:02x}{:02x}".format(*tuple(bgr)[::-1])


# The palette shared with the other vision-demos panels, read from config's
# PANEL_* (whose defaults are that palette), as in pull_ups. Hex is what
# matplotlib wants; _rgb() converts for the text drawing, so the plot and the
# bands around it cannot drift apart.
@dataclass(frozen=True)
class Palette:
    """The panel's colors as ``"#rrggbb"``."""

    bg: str
    fg: str
    muted: str
    grid: str
    full: str       # full depth, a straight body line
    partial: str    # cut short, hips off the line
    accent: str     # the per-rep line, the live dial short of depth, the depth bar

    @classmethod
    def from_cfg(cls, cfg) -> Palette:
        return cls(*(hex_from_bgr(getattr(cfg, f"PANEL_{f.name.upper()}")) for f in fields(cls)))


def _rgb(color: str) -> tuple[int, int, int]:
    """``"#rrggbb"`` -> ``(r, g, b)``."""
    return tuple(int(color[i:i + 2], 16) for i in (1, 3, 5))


def _plot_font(font):
    """Use the overlay's own face for the plot text, when matplotlib can load it.

    ``FontProperties(fname=...)`` reads face 0 of a .ttc. If it cannot be loaded
    we fall back to matplotlib's default rather than failing the render.
    """
    if font is None:
        return None
    try:
        from matplotlib import font_manager

        return font_manager.FontProperties(fname=font[0])
    except Exception:
        return None


def _grade_color(rep, c: Palette) -> str:
    return c.full if rep.full_depth else c.partial


def _verdict(reps, metric: str, c: Palette) -> list[tuple[str, tuple[int, int, int]]]:
    """The closing read: the best and worst rep on the graph, then the body line.

    Best and worst are colored by their own grade, like the dots they name, so
    the note and the graph read as one thing without a legend.
    """
    values = [rep_value(r, metric) for r in reps]
    best_i, worst_i = extremes(values)
    best_word, worst_word = (w.capitalize() for w in EXTREME_WORDS[metric])

    lines = [(f"{best_word}: rep {best_i + 1} ({format_value(values[best_i], metric)})",
              _rgb(_grade_color(reps[best_i], c)))]
    if len(reps) > 1 and values[worst_i] - values[best_i] > 1e-9:
        lines.append((f"{worst_word}: rep {worst_i + 1} "
                      f"({format_value(values[worst_i], metric)})",
                      _rgb(_grade_color(reps[worst_i], c))))
    else:
        lines.append(("even across the set", _rgb(c.muted)))

    note = body_line_note(reps, compact=True)
    flagged = any(r.body_line in ("sagging", "piked") for r in reps)
    lines.append((note, _rgb(c.partial if flagged else c.muted)))
    return lines


def _plot_image(analysis, shown: int, width: int, height: int, *, progressive: bool,
                font=None, y_ticks: int = 4, metric: str = "depth",
                title: str = "", ylabel: str = "", scale: float = 1.0, colors: Palette):
    """Render the per-rep scatter to a BGR array of exactly (height, width)."""
    c = colors
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    fp = _plot_font(font)

    def f(size: int) -> dict:
        """Font kwargs at *size*.

        A FontProperties carries its own size and matplotlib applies it *after*
        a plain ``fontsize=``, silently resetting the text to 10pt. Passing a
        sized copy is the only way both survive.
        """
        if fp is None:
            return {"fontsize": size}
        sized = fp.copy()
        sized.set_size(size)
        return {"fontproperties": sized}

    # Matplotlib sizes text in points, which become pixels via the dpi, so a
    # fixed dpi leaves the labels their 720px size on a 1662px panel. Scaling
    # the dpi makes one point the same fraction of the panel at any size.
    dpi = max(30.0, 100.0 * scale)
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi, facecolor=c.bg)
    ax = fig.add_subplot(111, facecolor=c.bg)

    reps = analysis.reps
    n = len(reps)
    xs = np.arange(1, n + 1)
    ys = np.array([rep_value(r, metric) for r in reps], dtype=float)
    seen = ys[np.isfinite(ys)]

    # The full-depth line is the grade itself, so on the depth graph each dot
    # lands on the same side of it as its color says.
    threshold = analysis.full_depth_degrees if metric == "depth" else None

    # Limits come from every rep, not just the visible ones, so the axes never
    # move as dots appear.
    if n:
        bounds = list(seen) + ([threshold] if threshold is not None else [])
        lo, hi = (min(bounds), max(bounds)) if bounds else (0.0, 1.0)
        span = max(hi - lo, 10.0 if metric == "depth" else 0.3)
        ax.set_ylim(lo - span * 0.55, hi + span * 0.55)
        ax.set_xlim(0.4, n + 0.6)
        ax.set_xticks(xs)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=y_ticks, prune=None))

    if threshold is not None and n:
        ax.axhline(threshold, color=c.muted, ls="--", lw=1.4, alpha=0.8, zorder=1)
        # Above the axes: a label beside the line would sit on a rep sooner or later.
        ax.annotate(f"- - full depth ≤ {threshold:.0f}°", (1, 1), xycoords="axes fraction",
                    xytext=(0, 5), textcoords="offset points", ha="right", va="bottom",
                    color=c.muted, **f(14))

    k = n if not progressive else max(0, min(shown, n))
    complete = k == n and n > 0
    best_i, worst_i = extremes(list(ys)) if n else (-1, -1)
    distinct = n > 1 and len(seen) > 1 and ys[worst_i] - ys[best_i] > 1e-9

    if k:
        ok = np.isfinite(ys[:k])
        ax.plot(xs[:k][ok], ys[:k][ok], color=c.accent, lw=2.4, alpha=0.55, zorder=2)
        ax.scatter(xs[:k][ok], ys[:k][ok], s=200, zorder=3,
                   c=[_grade_color(r, c) for r, keep in zip(reps[:k], ok) if keep],
                   edgecolors=c.bg, linewidths=2)

        best_word, worst_word = EXTREME_WORDS[metric]
        for j, (x, y) in enumerate(zip(xs[:k], ys[:k])):
            if not np.isfinite(y):
                continue
            ax.annotate(format_value(y, metric), (x, y), textcoords="offset points",
                        xytext=(0, 17), ha="center", color=c.fg, **f(17))
            # Best and worst only mean something once the set is over; during
            # the reveal "deepest so far" would move from dot to dot and mislead.
            # The best sits below its dot and the worst above its number, the
            # side the line leaves free.
            if complete and distinct and j in (best_i, worst_i):
                ax.annotate(best_word if j == best_i else worst_word, (x, y),
                            textcoords="offset points",
                            xytext=(0, -30) if j == best_i else (0, 36), ha="center",
                            color=c.muted, **f(15))

    ax.set_title(title, color=c.fg, pad=26 if threshold is not None else 20, **f(26))
    ax.set_xlabel("rep", color=c.muted, **f(18))
    ax.set_ylabel(ylabel, color=c.muted, **f(18))
    ax.tick_params(colors=c.muted, labelsize=16)
    if fp:
        ticks = fp.copy()
        ticks.set_size(16)
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontproperties(ticks)
    ax.grid(False)
    ax.set_box_aspect(1)
    for spine in ax.spines.values():
        spine.set_color(c.grid)

    fig.subplots_adjust(left=0.22, right=0.96, top=0.86, bottom=0.13)
    fig.canvas.draw()

    # Measured, not guessed: the header aligns to wherever matplotlib actually
    # laid the y tick labels, so the two stay aligned when the tick count, the
    # font, or the panel size changes.
    renderer = fig.canvas.get_renderer()
    fig_w, fig_h = fig.canvas.get_width_height()

    # The text keeps its size when the graph shrinks, so a fixed bottom margin
    # can leave the x label hanging off the figure. Raise the axes until it fits.
    low = ax.xaxis.label.get_window_extent(renderer).y0
    if low < 4:
        fig.subplots_adjust(bottom=0.13 + (4 - low) / fig_h)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()

    # The leftmost drawn thing, so the bands above and below share one visual
    # left edge with the axis.
    lefts = [t.get_window_extent(renderer).x0
             for t in ax.get_yticklabels() if t.get_text()]
    lefts.append(ax.yaxis.label.get_window_extent(renderer).x0)
    box = {"left": int(min(lefts)), "right": int(ax.get_window_extent(renderer).x1)}

    # Matplotlib measures from the bottom, rows count from the top. Flipping
    # here is what lets the caller center text against what is actually *seen*
    # rather than against the figure's outer edge, which is mostly padding.
    box["top"] = int(fig_h - ax.title.get_window_extent(renderer).y1)
    box["bottom"] = int(fig_h - ax.xaxis.label.get_window_extent(renderer).y0)

    rgba = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)

    bgr = rgba[..., 2::-1].copy()
    if bgr.shape[0] != height or bgr.shape[1] != width:
        import cv2
        ratio = width / bgr.shape[1]
        bgr = cv2.resize(bgr, (width, height))
        box = {k: int(v * ratio) for k, v in box.items()}
    return bgr, box


def _draw_lines(img, lines, font, *, px: int, x: int, gap: float = 0.40,
                max_width: int | None = None):
    """Draw ``(text, rgb)`` lines vertically centered in *img*, flush left at *x*.

    ``gap`` is the space between baselines as a multiple of the font size, so it
    scales with the text and the output resolution instead of being a pixel
    count that only looks right at one size. It is clamped so the block cannot
    outgrow its band. If the lines themselves are too tall or too wide for the
    band at *px*, the size steps down until they fit: smaller text beats text
    cut off at the edge of the band or run into the graph.

    Shared by the header and the closing note, so the two bands use identical
    metrics with no second implementation to drift out of step.
    """
    import cv2

    h, w = img.shape[:2]
    lines = [(t, c) for t, c in lines if t]
    if h < 20 or not lines:
        return img
    max_width = w - x if max_width is None else max_width
    n = len(lines)

    def layout(heights, size):
        """(gap_px, start_y) for a centered block that fits the band."""
        want = int(size * gap)
        if n > 1:
            room = (h - sum(heights)) / (n - 1)
            want = max(0, min(want, int(room)))
        block = sum(heights) + want * (n - 1)
        return want, max(0, (h - block) // 2)

    def fits(heights, widths, size):
        # A quarter of the size between lines at least, or they read as one block.
        return (sum(heights) + int(size * 0.25) * (n - 1) <= h
                and max(widths) <= max_width)

    if font is None:
        while True:
            fs = px / 34
            sizes = [cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)[0] for t, _ in lines]
            if px <= 8 or fits([s[1] for s in sizes], [s[0] for s in sizes], px):
                break
            px = int(px * 0.95)
        gap_px, y = layout([sz[1] for sz in sizes], px)
        for (text, rgb), (_, th) in zip(lines, sizes):
            y += th
            cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, fs,
                        rgb[::-1], 2, cv2.LINE_AA)
            y += gap_px
        return img

    from PIL import Image, ImageDraw, ImageFont

    pil = Image.fromarray(img[..., ::-1])
    draw = ImageDraw.Draw(pil)
    while True:
        face = ImageFont.truetype(font[0], px, index=font[1])
        boxes = [draw.textbbox((0, 0), text, font=face) for text, _ in lines]
        # One pitch for every line, from the face's own metrics, so a line with
        # a descender ("Push-ups") does not sit closer to the next than one without.
        ascent, descent = face.getmetrics()
        heights = [ascent + descent] * n
        if px <= 8 or fits(heights, [b[2] for b in boxes], px):
            break
        px = int(px * 0.95)
    gap_px, y = layout(heights, px)

    for (text, rgb), box, th in zip(lines, boxes, heights):
        draw.text((x - box[0], y), text, font=face, fill=rgb, anchor="la")
        y += th + gap_px

    img[:] = np.array(pil)[..., ::-1]
    return img


def credit_lines(cfg) -> list[tuple[str, str]]:
    """The credit as ``[(label, value), ...]``, top line first. Empty for none.

    Each ``CREDIT`` entry is a ``(label, value)`` pair or a plain string, as in
    deadlift. Blank entries are skipped, so an unfilled handle leaves no gap.
    """
    lines = []
    for entry in getattr(cfg, "CREDIT", None) or []:
        if isinstance(entry, (tuple, list)):
            label, value = (entry[0], entry[1]) if len(entry) >= 2 else ("", entry[0])
        else:
            label, value = "", entry
        label, value = str(label or "").strip(), str(value or "").strip()
        if value:
            lines.append((label, value))
    return lines


def _draw_credit(panel, font, *, cfg):
    """The credit stack, tucked into the bottom-right corner of the export.

    Each line is ``LABEL: value``, right-aligned, the label a shade dimmer. The
    last line's ink sits on the margin, so the gap on the right equals the gap
    underneath; the label shares the value's baseline, and the stack grows up.
    """
    import cv2

    lines = credit_lines(cfg)
    if not lines:
        return panel

    h, w = panel.shape[:2]
    px = max(9, int(cfg.CREDIT_SIZE * w / 720))
    pad = max(6, int(cfg.CREDIT_MARGIN * w / 720))
    gap = int(cfg.CREDIT_LINE_GAP * w / 720)

    def level(opacity):
        return int(255 * max(0.0, min(1.0, opacity)))

    value_a, label_a = level(cfg.CREDIT_OPACITY), level(cfg.CREDIT_LABEL_OPACITY)

    if font is None:
        y = h - pad
        for label, value in reversed(lines):
            text = f"{label}: {value}" if label else value
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, px / 34, 1)
            cv2.putText(panel, text, (w - pad - tw, y), cv2.FONT_HERSHEY_SIMPLEX,
                        px / 34, (value_a,) * 3, 1, cv2.LINE_AA)
            y -= th + gap
        return panel

    from PIL import Image, ImageDraw, ImageFont

    base = Image.fromarray(panel[..., ::-1]).convert("RGBA")
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    face = ImageFont.truetype(font[0], px, index=font[1])

    right = w - pad
    # Baseline of the last line, raised by its descent so its ink meets the margin.
    last = lines[-1]
    y = h - pad - max(draw.textbbox((0, 0), f"{last[0]}: {last[1]}" if last[0] else last[1],
                                    font=face, anchor="ls")[3], 0)
    for label, value in reversed(lines):
        draw.text((right, y), value, font=face, fill=(255, 255, 255, value_a), anchor="rs")
        if label:
            vw = draw.textlength(value, font=face)
            draw.text((right - vw, y), f"{label}: ", font=face,
                      fill=(255, 255, 255, label_a), anchor="rs")
        y -= px + gap

    panel[:] = np.array(Image.alpha_composite(base, layer).convert("RGB"))[..., ::-1]
    return panel


def _layout(width: int, height: int, *, cfg) -> dict:
    """Band heights and the graph's edge, in px, before anything is drawn.

    Every band is sized in px at a 720px-wide panel, like the text in it, so the
    stack is the same shape at any width. A taller panel (a portrait clip
    narrower than 9:16) spreads its spare height evenly between the bands; a
    shorter one takes it out of the graph, never out of a band of text.
    """
    u = width / 720
    head = int(cfg.PANEL_HEADER_BAND * u)
    live = int(cfg.PANEL_LIVE_BAND * u) if cfg.PANEL_LIVE else 0
    note = int(cfg.PANEL_NOTE_BAND * u)
    side = int(min(width, height) * cfg.PANEL_PLOT_SCALE)
    side = max(int(min(width, height) * 0.4), min(side, height - head - live - note))
    gap = max(0, (height - head - live - side - note) // 5)
    return {"head": head, "live": live, "note": note, "side": side, "gap": gap}


def build_panels(analysis, width: int, height: int, *, cfg) -> tuple[list, dict]:
    """One panel image per possible rep count, index 0..N, and where the live band goes."""
    font = resolve_font(cfg.PANEL_FONT, cfg.PANEL_FONT_INDEX)
    c = Palette.from_cfg(cfg)
    metric = cfg.REP_METRIC
    _, _, _, default_title, default_ylabel = METRICS[metric]

    # One square graph, centered across, with the header and the live block
    # stacked above it and the closing note below.
    lay = _layout(width, height, cfg=cfg)
    side, gap = lay["side"], lay["gap"]
    x0 = (width - side) // 2

    def plot(shown):
        return _plot_image(
            analysis, shown, side, side,
            progressive=cfg.PANEL_PROGRESSIVE, font=font, y_ticks=cfg.PANEL_Y_TICKS,
            metric=metric, title=cfg.PANEL_GRAPH_TITLE or default_title,
            ylabel=cfg.PANEL_Y_LABEL or default_ylabel, scale=width / 720, colors=c)

    # The figure's own padding above the title is not a band, so the graph is
    # placed by where its title actually starts, measured on the first render.
    first = plot(0)
    box = first[1]
    head_top = gap
    live_top = head_top + lay["head"] + gap
    y0 = live_top + lay["live"] + gap - box["top"]
    left = x0 + box["left"]
    note_band = y0 + box["bottom"]
    text_width = width - left - max(8, int(cfg.CREDIT_MARGIN * width / 720))

    panels = []
    for shown in range(len(analysis.reps) + 1):
        panel = np.zeros((height, width, 3), np.uint8)
        panel[:] = _rgb(c.bg)[::-1]

        image, box = first if shown == 0 else plot(shown)
        panel[y0:y0 + side, x0:x0 + side] = image

        done = analysis.reps[:shown] if cfg.PANEL_PROGRESSIVE else analysis.reps
        times = [r.total_seconds for r in done]
        full = sum(1 for r in done if r.full_depth)

        # Band 1: the headline numbers. The tally is green while every rep so
        # far reached depth, red once one did not.
        tally = c.muted if not done else c.full if full == len(done) else c.partial
        _draw_lines(panel[head_top:head_top + lay["head"]], [
            (f"{cfg.REP_LABEL}: {len(done)}", _rgb(c.fg)),
            (f"{cfg.PANEL_FULL_LABEL}: {full} / {len(done)}" if done
             else f"{cfg.PANEL_FULL_LABEL}: —", _rgb(tally)),
            (f"{cfg.PANEL_AVG_LABEL}: {np.mean(times):.2f}s" if times
             else f"{cfg.PANEL_AVG_LABEL}: —", _rgb(c.muted)),
            (f"{cfg.PANEL_LAST_LABEL}: {times[-1]:.2f}s" if times
             else f"{cfg.PANEL_LAST_LABEL}: —", _rgb(c.muted)),
        ], font, px=int(cfg.PANEL_HEADER_SIZE * width / 720), x=left,
           gap=cfg.PANEL_HEADER_LINE_GAP, max_width=text_width)

        # Band 2 is the live block, drawn per frame by LiveBlock.

        # Band 4: the closing note, once the set is actually over.
        if shown == len(analysis.reps) and analysis.reps:
            _draw_lines(panel[note_band:height - gap], _verdict(analysis.reps, metric, c), font,
                        px=int(cfg.PANEL_NOTE_SIZE * width / 720), x=left,
                        gap=cfg.PANEL_NOTE_LINE_GAP, max_width=text_width)

        _draw_credit(panel, font, cfg=cfg)
        panels.append(panel)

    live = {"top": live_top, "height": lay["live"], "left": left,
            "right": x0 + box["right"]}
    return panels, live


# ── The live block ───────────────────────────────────────────────────────────

PHASES = ("—", "going down", "pushing up", "resting at top")
_NONE, _DOWN, _UP, _REST = range(len(PHASES))

# The live body-line states, and the reps.py verdict each one reads as.
_LINE_NONE, _LINE_STRAIGHT, _LINE_HIGH, _LINE_SAG = range(4)
_LINE_VERDICTS = ("unreadable", "straight", "piked", "sagging")


def _held(state: np.ndarray, hold: int) -> np.ndarray:
    """*state*, changing only once a new value has lasted *hold* frames.

    Looks back only: frame i shows what has held up to frame i.
    """
    out = state.copy()
    for i in range(1, len(state)):
        lo = max(0, i - hold + 1)
        out[i] = state[i] if (state[lo:i + 1] == state[i]).all() else out[i - 1]
    return out


@lru_cache(maxsize=32)
def _face(path: str, index: int, px: int):
    from PIL import ImageFont

    return ImageFont.truetype(path, px, index=index)


def _text(img: np.ndarray, items, font) -> None:
    """Draw ``(text, (x, y), px, rgb, anchor)`` items; ``anchor`` is PIL's ("ls", "rs", "mb")."""
    if not items:
        return
    if font is None:
        for text, (x, y), px, rgb, anchor in items:
            fs = px / 34
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)
            x -= {"l": 0, "m": tw // 2, "r": tw}[anchor[0]]
            y += {"t": th, "m": th // 2, "b": 0, "s": 0}[anchor[1]]
            cv2.putText(img, text, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, fs,
                        rgb[::-1], 2, cv2.LINE_AA)
        return

    from PIL import Image, ImageDraw

    pil = Image.fromarray(img[..., ::-1])
    draw = ImageDraw.Draw(pil)
    for text, xy, px, rgb, anchor in items:
        draw.text(xy, text, font=_face(font[0], font[1], px), fill=rgb, anchor=anchor)
    img[:] = np.array(pil)[..., ::-1]


def _trailing_median(x: np.ndarray, k: int) -> np.ndarray:
    """Median of each value and the k-1 before it, ignoring NaNs.

    Steadier than the raw value and never ahead of the frame it is drawn on:
    it lags by about half the window, where a centered filter would read the
    future to stay on time.
    """
    if k < 2:
        return x
    pad = np.concatenate([np.full(k - 1, np.nan), x])
    windows = np.lib.stride_tricks.sliding_window_view(pad, k)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # an all-NaN window stays NaN
        return np.nanmedian(windows, axis=1)


def _running_percentile(x: np.ndarray, valid: np.ndarray, q: float) -> np.ndarray:
    """The *q*th percentile of the valid values of *x* up to and including each frame.

    NaN until the first valid value. Kept sorted as it goes, so each frame costs
    an insert rather than a sort, and frame i never reads frame i + 1.
    """
    out = np.full(len(x), np.nan)
    seen: list[float] = []
    for i, value in enumerate(x):
        if valid[i] and np.isfinite(value):
            insort(seen, float(value))
        if seen:
            pos = q / 100 * (len(seen) - 1)
            lo = int(pos)
            hi = min(lo + 1, len(seen) - 1)
            out[i] = seen[lo] + (seen[hi] - seen[lo]) * (pos - lo)
    return out


def _on_source_frames(analysis, series: np.ndarray, n_frames: int) -> np.ndarray:
    """*series*, indexed by returned frame, spread onto every source frame.

    A frame the model skipped holds the last value, as the overlay holds the
    last pose, so the block does not blink.
    """
    out = np.full(n_frames, np.nan)
    keep = analysis.indices < n_frames
    out[analysis.indices[keep]] = series[keep]
    last = np.maximum.accumulate(np.where(np.isfinite(out), np.arange(n_frames), -1))
    return np.where(last >= 0, out[np.maximum(last, 0)], np.nan)


class LiveBlock:
    """The band of the panel that moves every frame. One instance per render.

    Every series is worked out once, per source frame, in ``__init__``;
    ``draw`` only looks values up and paints them.

    - **Elbow angle**: the arm the per-rep grade reads (the one the model was
      surer of over that rep; between reps, over the clip), smoothed looking
      back only. At the bottom it settles on the number the rep is graded on.
    - **Phase**: straight from each rep's onset, bottom and lockout, so it
      always agrees with the per-rep table.
    - **Rep timer**: runs from the onset, holds the rep's total from lockout
      to the next onset.
    - **Body line**: the hips against the shoulder-to-feet line, on the side
      and foot reference each rep is graded on, smoothed looking back. A new
      verdict shows only once it has held for BODY_LINE_HOLD_SECONDS, the same
      rule the per-rep verdict uses, so it does not flicker at the cutoff.
    - **Depth**: measured on the body, not against other reps. 0% is the top
      plateau (arms locked), 100% is the shoulders down at the height of the
      hands. The plateau is the same percentile reps.py uses, taken over the
      plank frames up to this one, so the bar reads the same way on every rep,
      the first included, and never reads a later frame.
    """

    def __init__(self, analysis, n_frames: int, fps: float, width: int, *, top: int,
                 height: int, left: int, right: int, cfg, font):
        self.cfg, self.font = cfg, font
        self.colors = Palette.from_cfg(cfg)
        self.top, self.h, self.left, self.right = top, height, left, right
        self.w = width
        self.u = u = width / 720
        a = analysis
        self.sides = a.arm_sides
        n = max(n_frames, 1)
        frames = np.arange(n)

        in_plank = _on_source_frames(a, a.in_plank.astype(float), n) > 0.5
        first = a.reps[0].onset_frame if a.reps else n

        # Elbow angle on the arm each rep is graded on, the clip's clearer arm between reps.
        smooth = cfg.PANEL_LIVE_SMOOTH_FRAMES
        per_arm = [_on_source_frames(a, _trailing_median(arm, smooth), n) for arm in a.arm_raw]
        arm = np.full(n, a.arm_sides.index(a.elbow_side) if a.elbow_side in a.arm_sides else 0)
        for rep in a.reps:
            arm[rep.onset_frame:rep.lockout_frame + 1] = a.arm_sides.index(rep.elbow_side)
        self.arm = arm
        self.elbow = np.array([per_arm[s][i] for i, s in enumerate(arm)])
        self.elbow[~in_plank] = np.nan

        # Phase, from the same onset, bottom and lockout the table prints. Outside
        # a rep it is "resting at top" only while the shoulders are near the top:
        # a body lowered and left there is not resting at the top of anything.
        depth = _on_source_frames(a, a.depth, n)
        phase = np.full(n, _NONE)
        phase[(frames >= first) & (depth < cfg.REP_DEPTH_FRACTION * a.amplitude)] = _REST
        for rep in a.reps:
            phase[rep.onset_frame:rep.bottom_frame] = _DOWN
            phase[rep.bottom_frame:rep.lockout_frame] = _UP
        phase[~in_plank] = _NONE
        self.phase = phase

        # Rep timer: running from each onset, held at the total from lockout on.
        self.timer = np.full(n, np.nan)
        self.running = np.zeros(n, bool)
        self.rep_number = np.zeros(n, int)
        for k, rep in enumerate(a.reps):
            end = a.reps[k + 1].onset_frame if k + 1 < len(a.reps) else n
            span = frames[rep.onset_frame:end]
            running = span < rep.lockout_frame
            self.timer[span] = np.where(running, span / fps - rep.onset_time, rep.total_seconds)
            self.running[span] = running
            self.rep_number[span] = rep.number

        # Body line: each rep's own side and foot reference from its onset to the
        # next one, the clip's elsewhere, held before it changes.
        lines = {ref: [_on_source_frames(a, _trailing_median(side, smooth), n)
                       for side in per_side] for ref, per_side in a.hip_lines.items()}
        line = _on_source_frames(a, _trailing_median(a.hip_line, smooth), n)
        for k, rep in enumerate(a.reps):
            end = a.reps[k + 1].onset_frame if k + 1 < len(a.reps) else n
            span = slice(rep.onset_frame, end)
            line[span] = lines[rep.hip_reference][a.body_sides.index(rep.body_side)][span]
        tol = cfg.BODY_LINE_TOLERANCE_DEG
        state = np.where(~np.isfinite(line), _LINE_NONE,
                         np.where(np.abs(line) <= tol, _LINE_STRAIGHT,
                                  np.where(line < 0, _LINE_HIGH, _LINE_SAG)))
        state[~in_plank] = _LINE_NONE
        self.line = line
        self.line_state = _held(state, max(1, int(round(cfg.BODY_LINE_HOLD_SECONDS * fps))))

        # Depth: the shoulders' height above the wrists, smoothed looking back.
        # The hands are the floor of the movement, so the top plateau is 0% and
        # the shoulders level with the hands are 100%, on every rep alike.
        rise = _on_source_frames(a, _trailing_median(a.raw, smooth), n)
        top = _running_percentile(rise, in_plank, cfg.TOP_PERCENTILE)
        with np.errstate(invalid="ignore", divide="ignore"):
            share = np.clip((top - rise) / top, 0.0, 1.0)
        share[~(top > 0) | ~in_plank | (frames < first)] = np.nan
        self.depth = share

        self.base = self._background()

    def _s(self, px: float) -> int:
        """*px* at a 720px-wide panel, at this panel's width."""
        return max(1, int(round(px * self.u)))

    # Positions inside the band, px at a 720px-wide panel, top-down.
    _CAPTION_Y = 4      # the "Elbow" caption's top
    _NUMBER_Y = 112     # the live angle's baseline
    _GAUGE_R = 62       # the dial's radius; its center sits on the number's baseline
    _PHASE_Y = 170      # phase and timer baselines
    _LINE_Y = 214       # the body-line caption and readout
    _BAR_LABEL_Y = 254  # the depth caption's baseline
    _BAR_Y = 266        # the depth bar's top
    _BAR_H = 18

    def _gauge_center(self) -> tuple[int, int]:
        r = self._s(self._GAUGE_R)
        return self.right - r - self._s(6), self._s(self._NUMBER_Y)

    def _background(self) -> np.ndarray:
        """What never moves: the depth caption, the dial's track and 90° mark, the bar's track."""
        cfg, s, c = self.cfg, self._s, self.colors
        band = np.zeros((self.h, self.w, 3), np.uint8)
        band[:] = _rgb(c.bg)[::-1]

        (cx, cy), r = self._gauge_center(), s(self._GAUGE_R)
        thick = s(10)
        # The dial is the elbow itself: the upper arm points right from the center,
        # the forearm sweeps from straight (left, 180°) toward the shoulder.
        cv2.ellipse(band, (cx, cy), (r, r), 0, 180, 360, _rgb(c.grid)[::-1], thick, cv2.LINE_AA)
        mark = np.radians(cfg.FULL_DEPTH_ELBOW_DEG)
        inner, outer = r - thick, r + thick
        p0 = (int(cx + inner * np.cos(np.pi - mark)), int(cy - inner * np.sin(np.pi - mark)))
        p1 = (int(cx + outer * np.cos(np.pi - mark)), int(cy - outer * np.sin(np.pi - mark)))
        cv2.line(band, p0, p1, _rgb(c.fg)[::-1], max(1, s(3)), cv2.LINE_AA)

        x0, x1 = self.left, self.right
        y0 = s(self._BAR_Y)
        cv2.rectangle(band, (x0, y0), (x1, y0 + s(self._BAR_H)), _rgb(c.grid)[::-1], -1)

        label = (int(cx + (outer + s(4)) * np.cos(np.pi - mark)),
                 int(cy - (outer + s(4)) * np.sin(np.pi - mark)))
        _text(band, [
            (f"{cfg.FULL_DEPTH_ELBOW_DEG:.0f}°", label, s(16), _rgb(c.muted), "mb"),
            (cfg.PANEL_LIVE_LINE_LABEL, (x0, s(self._LINE_Y)),
             s(cfg.PANEL_LIVE_CAPTION_SIZE), _rgb(c.muted), "ls"),
            (cfg.PANEL_LIVE_DEPTH_LABEL, (x0, s(self._BAR_LABEL_Y)),
             s(cfg.PANEL_LIVE_CAPTION_SIZE), _rgb(c.muted), "ls"),
        ], self.font)
        return band

    def draw(self, frame: int) -> np.ndarray:
        """The band for *frame*, ready to paste at ``self.top``."""
        cfg, s, c = self.cfg, self._s, self.colors
        i = int(np.clip(frame, 0, len(self.phase) - 1))
        band = self.base.copy()
        items = []

        # Elbow: the number, and the dial filled from straight to where the arm is.
        angle = self.elbow[i]
        deep = bool(np.isfinite(angle) and round(angle) <= cfg.FULL_DEPTH_ELBOW_DEG)
        color = c.full if deep else c.accent
        items.append((f"{cfg.PANEL_LIVE_ELBOW_LABEL}, {self.sides[self.arm[i]]} arm",
                      (self.left, s(self._CAPTION_Y)), s(cfg.PANEL_LIVE_CAPTION_SIZE),
                      _rgb(c.muted), "lt"))
        items.append((f"{angle:.0f}°" if np.isfinite(angle) else "—",
                      (self.left, s(self._NUMBER_Y)), s(cfg.PANEL_LIVE_NUMBER_SIZE),
                      _rgb(c.full if deep else c.fg) if np.isfinite(angle) else _rgb(c.muted),
                      "ls"))
        if np.isfinite(angle):
            (cx, cy), r = self._gauge_center(), s(self._GAUGE_R)
            reach = float(np.clip(angle, 0, 180))
            cv2.ellipse(band, (cx, cy), (r, r), 0, 180, 360 - reach, _rgb(color)[::-1],
                        s(10), cv2.LINE_AA)

        # Phase and the rep timer, one line: what is happening, and for how long.
        phase = self.phase[i]
        items.append((PHASES[phase], (self.left, s(self._PHASE_Y)), s(cfg.PANEL_LIVE_SIZE),
                      _rgb(c.muted if phase in (_NONE, _REST) else c.fg), "ls"))
        t = self.timer[i]
        timer = (f"{cfg.PANEL_LIVE_TIMER_LABEL} {self.rep_number[i]}: {t:.2f}s"
                 if np.isfinite(t) else f"{cfg.PANEL_LIVE_TIMER_LABEL}: —")
        items.append((timer, (self.right, s(self._PHASE_Y)), s(cfg.PANEL_LIVE_SIZE),
                      _rgb(c.fg if self.running[i] else c.muted), "rs"))

        # Body line, colored like the grades: green straight, red off the line.
        state, dev = self.line_state[i], self.line[i]
        verdict = BODY_LINE_LABELS[_LINE_VERDICTS[state]]
        if state in (_LINE_HIGH, _LINE_SAG) and np.isfinite(dev):
            verdict += f" {abs(dev):.0f}°"
        items.append(("—" if state == _LINE_NONE else verdict,
                      (self.right, s(self._LINE_Y)), s(cfg.PANEL_LIVE_LINE_SIZE),
                      _rgb(c.muted if state == _LINE_NONE else
                           c.full if state == _LINE_STRAIGHT else c.partial), "rs"))

        # Depth: filled from the top plateau (empty) toward the hands (full).
        share = self.depth[i]
        y0 = s(self._BAR_Y)
        if np.isfinite(share):
            x1 = self.left + int(round(share * (self.right - self.left)))
            if x1 > self.left:
                cv2.rectangle(band, (self.left, y0), (x1, y0 + s(self._BAR_H)),
                              _rgb(c.accent)[::-1], -1)
        items.append((f"{share * 100:.0f}%" if np.isfinite(share) else "—",
                      (self.right, s(self._BAR_LABEL_Y)), s(cfg.PANEL_LIVE_CAPTION_SIZE),
                      _rgb(c.fg if np.isfinite(share) else c.muted), "rs"))

        _text(band, items, self.font)
        return band


class SidePanel:
    """The whole panel for any frame: the per-rep image for that count, plus the live band."""

    def __init__(self, analysis, n_frames: int, fps: float, width: int, height: int, *, cfg):
        self.analysis = analysis
        self.panels, band = build_panels(analysis, width, height, cfg=cfg)
        self.live = None
        if cfg.PANEL_LIVE and band["height"] > 0:
            font = resolve_font(cfg.PANEL_FONT, cfg.PANEL_FONT_INDEX)
            self.live = LiveBlock(analysis, n_frames, fps, width, cfg=cfg, font=font, **band)

    @property
    def states(self) -> int:
        return len(self.panels)

    def draw(self, frame: int) -> np.ndarray:
        panel = self.panels[min(self.analysis.count_at(frame), len(self.panels) - 1)]
        if self.live is None:
            return panel
        panel = panel.copy()
        top = self.live.top
        panel[top:top + self.live.h] = self.live.draw(frame)
        return panel
