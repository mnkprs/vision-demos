"""The side panel: rep count, vertical displacement over time, and a row per rep.

All reps are known up front, so the static parts (axes, headings, credit) are drawn
once and copied per frame; fixed axes mean the curve fills in without rescaling.
"""

from __future__ import annotations

from functools import lru_cache
from typing import NamedTuple

import cv2
import numpy as np

from .reps import rep_value
from .text import resolve_font

class Palette(NamedTuple):
    """The panel's colors as ``"#rrggbb"``, from config's ``PANEL_*`` (BGR there)."""

    bg: str
    fg: str
    muted: str
    grid: str
    full: str      # full range of motion
    partial: str   # cut short
    accent: str    # the displacement line and its playhead

    @classmethod
    def from_cfg(cls, cfg) -> "Palette":
        return cls(*("#{:02x}{:02x}{:02x}".format(*getattr(cfg, f"PANEL_{name.upper()}")[::-1])
                     for name in cls._fields))


def _rgb(color: str) -> tuple[int, int, int]:
    """``"#rrggbb"`` -> ``(r, g, b)``."""
    return tuple(int(color[i:i + 2], 16) for i in (1, 3, 5))


def _bgr(color: str) -> tuple[int, int, int]:
    return _rgb(color)[::-1]


@lru_cache(maxsize=32)
def _face(path: str, index: int, px: int):
    from PIL import ImageFont

    return ImageFont.truetype(path, px, index=index)


def _text(img: np.ndarray, items, font) -> np.ndarray:
    """Draw ``(text, (x, y), px, rgb, anchor)`` items; ``anchor`` is PIL's (e.g. "lm", "rm")."""
    if not items:
        return img
    if font is None:
        for text, (x, y), px, rgb, anchor in items:
            fs = px / 34
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)
            x -= {"l": 0, "m": tw // 2, "r": tw}[anchor[0]]
            y += {"t": th, "m": th // 2, "b": 0, "s": 0}[anchor[1]]
            cv2.putText(img, text, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, fs,
                        rgb[::-1], 2, cv2.LINE_AA)
        return img

    from PIL import Image, ImageDraw

    pil = Image.fromarray(img[..., ::-1])
    draw = ImageDraw.Draw(pil)
    for text, xy, px, rgb, anchor in items:
        draw.text(xy, text, font=_face(font[0], font[1], px), fill=rgb, anchor=anchor)
    img[:] = np.array(pil)[..., ::-1]
    return img


def _text_width(text: str, px: int, font) -> int:
    """Rendered width of *text* at *px*, in the same face `_text` draws with."""
    if font is None:
        return cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, px / 34, 2)[0][0]
    return int(_face(font[0], font[1], px).getlength(text))


def _axes_image(width: int, height: int, *, duration: float, ylim, yticks, yfmt,
                title: str, xlabel: str, ylabel: str, scale: float, font, margin: int,
                colors: Palette, threshold=None, threshold_label: str = "", hlines=()):
    """Static axes as BGR, plus the axes' pixel box ``(x0, y0, x1, y1)``, top-down."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, MultipleLocator

    fp = None
    if font is not None:
        try:
            from matplotlib import font_manager
            fp = font_manager.FontProperties(fname=font[0])
        except Exception:
            fp = None

    def f(size: float) -> dict:
        # A FontProperties overrides `fontsize=`, so pass a sized copy instead.
        if fp is None:
            return {"fontsize": size}
        sized = fp.copy()
        sized.set_size(size)
        return {"fontproperties": sized}

    dpi = max(30.0, 100.0 * scale)
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi, facecolor=colors.bg)
    ax = fig.add_subplot(111, facecolor=colors.bg)
    ax.set_xlim(0, duration)
    ax.set_ylim(*ylim)
    ax.set_yticks(yticks)
    ax.yaxis.set_major_formatter(FuncFormatter(yfmt))
    # Whole seconds: a tick every 5 s, or every 1 or 2 s on a short clip.
    ax.xaxis.set_major_locator(MultipleLocator(5 if duration > 12 else 2 if duration > 5 else 1))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}"))

    for y in hlines:
        ax.axhline(y, color=colors.muted, ls="--", lw=1.4, alpha=0.8)
    if threshold is not None:
        ax.axhline(threshold, color=colors.muted, ls="--", lw=1.4, alpha=0.8)
        if threshold_label:
            # Above the axes: a label beside the line would be drawn over every rep.
            ax.annotate(f"- - {threshold_label}", (1, 1), xycoords="axes fraction",
                        xytext=(0, 4), textcoords="offset points", ha="right", va="bottom",
                        color=colors.muted, **f(12))

    ax.set_title(title, color=colors.fg, pad=12, **f(21))
    ax.set_xlabel(xlabel, color=colors.muted, labelpad=4, **f(15))
    ax.set_ylabel(ylabel, color=colors.muted, labelpad=6, **f(15))
    ax.tick_params(colors=colors.muted, labelsize=13)
    if fp:
        ticks = fp.copy()
        ticks.set_size(13)
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontproperties(ticks)
    for spine in ax.spines.values():
        spine.set_color(colors.grid)

    # The right edge meets the panel margin, so the axes end where the text does.
    # Wider on the left when there's a y label beside the tick labels.
    fig.subplots_adjust(left=0.22 if ylabel else 0.17, right=1 - margin / width,
                        top=0.90, bottom=0.13)
    fig.canvas.draw()

    renderer = fig.canvas.get_renderer()
    fig_h = fig.canvas.get_width_height()[1]
    ext = ax.get_window_extent(renderer)
    box = (int(ext.x0), int(fig_h - ext.y1), int(ext.x1), int(fig_h - ext.y0))

    rgba = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)
    bgr = rgba[..., 2::-1].copy()
    if bgr.shape[:2] != (height, width):
        rx, ry = width / bgr.shape[1], height / bgr.shape[0]
        bgr = cv2.resize(bgr, (width, height))
        box = (int(box[0] * rx), int(box[1] * ry), int(box[2] * rx), int(box[3] * ry))
    return bgr, box


def display_smooth(values: np.ndarray, fps: float, seconds: float, *,
                   bridge_seconds: float = 0.1) -> np.ndarray:
    """Centered Gaussian smoothing for display; *seconds* is the sigma.

    NaN gaps up to *bridge_seconds* are bridged; longer ones (hands off the bar) stay gaps.
    """
    out = values.astype(float).copy()
    finite = np.isfinite(out)
    if finite.sum() < 2:
        return out
    idx = np.arange(len(out))
    filled = np.interp(idx, idx[finite], out[finite])

    # Which NaN runs are short enough to bridge.
    keep = finite.copy()
    limit = max(1, int(round(bridge_seconds * fps)))
    i = 0
    while i < len(out):
        if finite[i]:
            i += 1
            continue
        j = i
        while j < len(out) and not finite[j]:
            j += 1
        if i > 0 and j < len(out) and j - i <= limit:
            keep[i:j] = True
        i = j

    sigma = seconds * fps
    if sigma >= 0.5 and len(filled) >= 3:
        half = int(3 * sigma)
        kernel = np.exp(-0.5 * (np.arange(-half, half + 1) / sigma) ** 2)
        kernel /= kernel.sum()
        filled = np.convolve(np.pad(filled, half, mode="edge"), kernel, mode="valid")
    return np.where(keep, filled, np.nan)


class _Strip:
    """One time-series graph: its axes box and a mapping from values to panel pixels."""

    def __init__(self, base: np.ndarray, top: int, height: int, n: int, *,
                 ylim, scale: float, font, margin: int, colors: Palette, **axes_kw):
        width = base.shape[1]
        img, (x0, y0, x1, y1) = _axes_image(width, height, ylim=ylim, scale=scale, font=font,
                                            margin=margin, colors=colors, **axes_kw)
        self.colors = colors
        base[top:top + height] = img
        self.box = (x0, top + y0, x1, top + y1)
        x0, _, x1, _ = self.box
        self.lo, self.hi = ylim
        self.px = (x0 + np.arange(n) / max(n, 1) * (x1 - x0)).astype(np.int32)

    def y_of(self, value):
        """Panel row of *value*, NaN where the value is NaN. Works on a flipped axis too."""
        _, y0, _, y1 = self.box
        clipped = np.clip(value, min(self.lo, self.hi), max(self.lo, self.hi))
        return y1 - (clipped - self.lo) / (self.hi - self.lo) * (y1 - y0)

    def shade(self, panel, a: int, b: int, color: str, alpha: float = 0.16) -> None:
        """Tint the frames ``[a, b]`` across the full height of the axes."""
        _, y0, _, y1 = self.box
        if b <= a:
            return
        xa = int(self.px[min(a, len(self.px) - 1)])
        xb = int(self.px[min(b, len(self.px) - 1)])
        roi = panel[y0:y1, xa:xb + 1]
        roi[:] = (roi * (1 - alpha) + np.array(_bgr(color)) * alpha).astype(np.uint8)

    def line(self, panel, py: np.ndarray, k: int, color: str, thick: int,
             alpha: float = 1.0) -> None:
        """The curve up to frame *k*, broken wherever the value is NaN."""
        k = min(k, len(self.px))
        if k < 2:
            return
        ys = py[:k]
        ok = np.isfinite(ys)
        runs, start = [], None
        for i in range(k + 1):
            if i < k and ok[i]:
                start = i if start is None else start
            elif start is not None:
                if i - start >= 2:
                    runs.append(np.stack([self.px[start:i], ys[start:i].astype(np.int32)], 1)
                                .reshape(-1, 1, 2))
                start = None
        if not runs:
            return
        if alpha >= 1:
            cv2.polylines(panel, runs, False, _bgr(color), thick, cv2.LINE_AA)
            return
        # Blended inside the axes only, so a faint line costs a strip, not a panel.
        x0, y0, x1, y1 = self.box
        pad = thick + 2
        ya, yb = max(0, y0 - pad), min(panel.shape[0], y1 + pad)
        roi = panel[ya:yb, x0:x1 + 1]
        layer = roi.copy()
        cv2.polylines(layer, [r - [x0, ya] for r in runs], False, _bgr(color), thick,
                      cv2.LINE_AA)
        cv2.addWeighted(layer, alpha, roi, 1 - alpha, 0, dst=roi)

    def playhead(self, panel, k: int, scale: float) -> None:
        _, y0, _, y1 = self.box
        px = int(self.px[min(max(k - 1, 0), len(self.px) - 1)])
        cv2.line(panel, (px, y0), (px, y1), _bgr(self.colors.muted), max(1, int(1.5 * scale)),
                 cv2.LINE_AA)

    def dot(self, panel, frame: int, py: float, color: str, scale: float, r: float = 7) -> None:
        if not np.isfinite(py):
            return
        px = int(self.px[min(max(frame, 0), len(self.px) - 1)])
        cv2.circle(panel, (px, int(py)), int((r + 2) * scale), _bgr(self.colors.bg), -1,
                   cv2.LINE_AA)
        cv2.circle(panel, (px, int(py)), int(r * scale), _bgr(color), -1, cv2.LINE_AA)


class Panel:
    """Builds the panel for any frame from one static base."""

    def __init__(self, analysis, n_frames: int, fps: float, width: int, height: int, *, cfg):
        self.analysis, self.cfg = analysis, cfg
        self.w, self.h = width, height
        s = self.s = width / 720
        self.font = resolve_font(cfg.PANEL_FONT, cfg.PANEL_FONT_INDEX)
        c = self.colors = Palette.from_cfg(cfg)
        self.margin = int(48 * s)

        # Per source frame, blanked while the hands are off the bar, smoothed for display.
        on_bar = analysis.on_bar if len(analysis.on_bar) == len(analysis.indices) \
            else np.ones(len(analysis.indices), bool)
        def shown(series):
            return display_smooth(
                analysis.on_source_frames(np.where(on_bar, series, np.nan), n_frames),
                fps, cfg.PANEL_ANGLE_SMOOTH_SECONDS)

        self.mean = shown(analysis.elbow)   # the live elbow value

        # Layout as fractions of panel height. Rep rows share the space above the
        # credit, so short sets spread out and long ones pack tighter.
        self.y_reps = int(height * 0.045)
        self.y_live = int(height * 0.098)
        self.y_rows = int(height * 0.625)
        rows_bottom = int(height * 0.915)
        n_rows = max(len(analysis.reps), 3)
        span = max(0, rows_bottom - self.y_rows)
        # Never closer than a line of text; a long set shows its newest reps instead.
        min_gap = int(cfg.PANEL_ROW_SIZE * s * 1.3)
        self.row_gap = max(1, min_gap, min(int(height * 0.06), span // n_rows))
        self.room = max(1, span // self.row_gap)

        base = np.zeros((height, width, 3), np.uint8)
        base[:] = _bgr(c.bg)

        # Head above hands, in torso lengths: the quantity the top is graded on, so
        # "clears bar" is the real cutoff. The y-axis marks only the two meaningful
        # heights (dead hang, clears bar), by name: torso lengths aren't a unit anyone
        # reads. Headroom holds rep labels.
        self.height = shown(analysis.clearance)
        seen = self.height[np.isfinite(self.height)]
        hang = float(np.percentile(seen, cfg.BASELINE_PERCENTILE)) if len(seen) else 0.0
        clears = analysis.clearance_required
        top = max(float(seen.max()) if len(seen) else hang + 0.1,
                  clears if np.isfinite(clears) else hang)
        bottom = min(float(seen.min()) if len(seen) else hang, hang)
        span = max(top - bottom, 1e-3)
        marks = {hang: "dead hang"} | ({clears: "clears bar"} if np.isfinite(clears) else {})
        self.strip = _Strip(
            base, int(height * 0.135), int(height * 0.44), n_frames,
            ylim=(bottom - span * 0.08, top + span * 0.22), yticks=sorted(marks),
            yfmt=lambda v, _: min(marks.items(), key=lambda m: abs(m[0] - v))[1],
            duration=n_frames / fps, title="Vertical displacement", xlabel="Time (s)",
            ylabel="Head height", hlines=sorted(marks), scale=s, font=self.font,
            margin=self.margin, colors=c)
        self.height_py = self.strip.y_of(self.height)

        # Column centers for REP | ROM | BOTTOM | TOP | PULL, shared by heading and rows.
        content = width - 2 * self.margin
        self.cols = tuple(int(self.margin + content * f) for f in (0.1, 0.3, 0.5, 0.7, 0.9))

        # ROM tally left, elbow angle right; shrunk until the longest values never meet.
        px = int(cfg.PANEL_LIVE_SIZE * s)
        while px > 10 and (_text_width("Full ROM: 88 / 88", px, self.font)
                           + _text_width("Elbow 180°", int(px * 0.8), self.font)
                           + int(32 * s)) > content:
            px = int(px * 0.95)
        self.live_px = px

        muted, note_px = _rgb(c.muted), int(cfg.PANEL_NOTE_SIZE * s)
        y = self.y_rows
        _text(base, [(head, (x, y), note_px, muted, "mm")
                     for head, x in zip(("REP", "ROM", "BOTTOM", "TOP", "PULL"), self.cols)],
              self.font)
        rule_y = self.y_rows + int(18 * s)
        cv2.line(base, (self.margin, rule_y), (width - self.margin, rule_y), _bgr(c.grid),
                 max(1, int(s)))
        _draw_credit(base, self.font, cfg=cfg, scale=s)
        self.base = base

    def verdict_color(self, rep) -> str:
        return self.colors.full if rep.complete else self.colors.partial

    def render(self, frame: int) -> np.ndarray:
        s, cfg, c = self.s, self.cfg, self.colors
        panel = self.base.copy()
        done = self.analysis.done_at(frame)
        k = frame + 1
        thick = max(2, int(3 * s))
        strip = self.strip
        _, y0, _, _ = strip.box

        # Finished reps: shaded by verdict, numbered at the peak, graded points dotted.
        # A number that would touch the one before it is skipped, so dense reps stay legible.
        items = []
        label_px, label_right = int(18 * s), -1
        for rep in done:
            color = self.verdict_color(rep)
            strip.shade(panel, rep.onset_frame, min(rep.end_frame, frame), color)
            x = int(strip.px[min(rep.peak_frame, len(strip.px) - 1)])
            half = _text_width(str(rep.number), label_px, self.font) // 2
            if x - half > label_right:
                items.append((str(rep.number), (x, y0 + int(14 * s)), label_px, _rgb(color), "mm"))
                label_right = x + half + int(6 * s)
        strip.line(panel, self.height_py, k, c.accent, thick + max(1, int(s)))
        strip.playhead(panel, k, s)
        for rep in done:
            if rep.rom is None:
                continue
            g = rep.rom
            strip.dot(panel, rep.valley_frame, self.height_py[rep.valley_frame],
                      c.full if g.bottom_ok else c.partial, s, r=5)
            strip.dot(panel, rep.peak_frame, self.height_py[rep.peak_frame],
                      c.full if g.top_ok else c.partial, s, r=5)
        i = min(frame, len(self.height_py) - 1)
        strip.dot(panel, i, self.height_py[i], c.accent, s)

        # Header: rep count and average pull, then ROM tally and live elbow angle.
        x1 = self.w - self.margin
        pulls = [rep_value(r, cfg.REP_METRIC) for r in done if r.complete]
        graded = [r for r in done if r.rom is not None]
        full = sum(1 for r in graded if r.complete)
        head_px = int(cfg.PANEL_HEADER_SIZE * s)
        tally_color = c.muted if not graded else c.full if full == len(graded) else c.partial
        items += [
            (f"Reps: {len(done)}", (self.margin, self.y_reps), head_px, _rgb(c.fg), "lm"),
            (f"Avg pull: {np.mean(pulls):.2f}s" if pulls else "Avg pull: —",
             (x1, self.y_reps), int(head_px * 0.7), _rgb(c.muted), "rm"),
            (f"Full ROM: {full} / {len(graded)}" if graded else "Full ROM: —",
             (self.margin, self.y_live), self.live_px, _rgb(tally_color), "lm"),
        ]
        elbow = self.mean[min(frame, len(self.mean) - 1)] if len(self.mean) else np.nan
        items.append((f"Elbow {elbow:.0f}°" if np.isfinite(elbow) else "Elbow —",
                      (x1, self.y_live), int(self.live_px * 0.8), _rgb(c.fg), "rm"))

        # One row per rep after its peak; the newest win if rows run out.
        row_px = int(cfg.PANEL_ROW_SIZE * s)
        for j, rep in enumerate(done[-self.room:]):
            y = self.y_rows + (j + 1) * self.row_gap
            g = rep.rom

            def angle(value, ok):
                if g is None or not np.isfinite(value):
                    return "—", _rgb(c.muted)
                return f"{value:.0f}°", _rgb(c.fg if ok else c.partial)

            bottom = angle(g.hang_degrees, g.bottom_ok) if g else ("—", _rgb(c.muted))
            top = angle(g.peak_degrees, g.top_ok) if g else ("—", _rgb(c.muted))
            verdict = (("FULL" if rep.complete else "PARTIAL"), _rgb(self.verdict_color(rep))) \
                if g else ("—", _rgb(c.muted))
            # Partial reps are not timed: a short pull is not a pull-up time.
            pull = (f"{rep_value(rep, cfg.REP_METRIC):.2f}s", _rgb(c.fg)) if rep.complete \
                else ("—", _rgb(c.muted))
            cells = [(str(rep.number), _rgb(c.fg)), verdict, bottom, top, pull]
            items += [(text, (x, y), row_px, color, "mm")
                      for (text, color), x in zip(cells, self.cols)]
        _text(panel, items, self.font)
        return panel


def credit_lines(cfg) -> list[tuple[str, str]]:
    """``cfg.CREDIT`` as ``[(label, value), ...]``, top line first; blank entries skipped.

    Each entry is a ``(label, value)`` pair or a plain string.
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


def _draw_credit(panel, font, *, cfg, scale: float) -> None:
    """Right-aligned ``LABEL: value`` lines stacked upward from the bottom-right corner."""
    lines = credit_lines(cfg)
    if not lines:
        return
    h, w = panel.shape[:2]
    px = max(9, int(cfg.CREDIT_SIZE * scale))
    gap = int(cfg.CREDIT_LINE_GAP * scale)
    right = w - int(cfg.CREDIT_MARGIN * scale)
    y = h - int(cfg.CREDIT_MARGIN * scale)
    value_rgb, label_rgb = cfg.CREDIT_COLOR[::-1], cfg.CREDIT_LABEL_COLOR[::-1]

    if font is None:
        for label, value in reversed(lines):
            text = f"{label}: {value}" if label else value
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, px / 34, 1)
            cv2.putText(panel, text, (right - tw, y), cv2.FONT_HERSHEY_SIMPLEX, px / 34,
                        cfg.CREDIT_COLOR, 1, cv2.LINE_AA)
            y -= th + gap
        return

    from PIL import Image, ImageDraw

    pil = Image.fromarray(panel[..., ::-1])
    draw = ImageDraw.Draw(pil)
    face = _face(font[0], font[1], px)
    for label, value in reversed(lines):
        # Baseline anchors ("rs"), so the label and value sit on one line.
        draw.text((right, y), value, font=face, fill=value_rgb, anchor="rs")
        if label:
            vw = draw.textlength(value, font=face)
            draw.text((right - vw, y), f"{label}: ", font=face, fill=label_rgb, anchor="rs")
        y -= px + gap
    panel[:] = np.array(pil)[..., ::-1]
