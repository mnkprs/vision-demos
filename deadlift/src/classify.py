"""Per-frame back-position reads through TypeSafe System One on the VLM Run Gateway.

One read is one `choice` question over one frame. System One does not parse a
free-text answer: the reply is constrained to the question's labels and the
model's own probability for each label is read off at the answer position, so
every read returns exactly those probabilities and nothing else, in about half a
second. That is what makes it cheap enough to ask the same question of every
frame.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn

from .video import file_sha256

QUESTION_ID = "back"


@dataclass(frozen=True)
class Read:
    """One answered frame."""

    index: int                        # source frame index
    time: float                       # seconds into the clip
    choice: str | None
    probabilities: dict[str, float]
    confidence: float | None
    latency: float                    # client-measured round trip, seconds
    usage: dict
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


def sample_indices(n_frames: int, fps: float, sample_fps: float | None) -> list[int]:
    """Frame indices to read, evenly spaced at *sample_fps* (every frame when None)."""
    if not sample_fps or sample_fps >= fps:
        return list(range(n_frames))
    step = fps / sample_fps
    return sorted({min(n_frames - 1, round(i * step)) for i in range(int(n_frames / step) + 1)})


def crop_box(w: int, h: int, crop) -> tuple[int, int, int, int]:
    """Pixel ``(x0, y0, x1, y1)`` of the normalized *crop*, or the whole frame."""
    if crop is None:
        return 0, 0, w, h
    x0, y0, x1, y1 = crop
    return int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)


def encode_frame(frame: np.ndarray, *, long_edge: int, crop, quality: int) -> str:
    """Crop, downscale and JPEG-encode one BGR frame as a ``data:`` URL."""
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = crop_box(w, h, crop)
    img = frame[y0:y1, x0:x1]
    scale = long_edge / max(img.shape[:2])
    if scale < 1:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def person_crops(by_index: dict[int, list[dict]], indices: list[int], *, pad: float
                 ) -> dict[int, tuple[float, float, float, float]]:
    """The ViTPose person box around each sampled frame, padded, as normalized ``(x0, y0, x1, y1)``.

    The lifter fills a small part of a wide shot, and a read spends a fixed
    vision budget on whatever image it gets, so cropping to the person spends it
    on the back instead of the gym. The pad is a share of the box on every side:
    ViTPose's box is tight to the joints, and a rounded upper back bulges past
    the line from shoulder to hip.

    A frame with nobody detected borrows the nearest frame that has someone, so
    the crop never jumps to the full frame for one read.
    """
    posed = sorted(i for i, persons in by_index.items() if persons)
    crops = {}
    for i in indices:
        if not posed:
            break
        j = min(posed, key=lambda k: abs(k - i))
        x, y, w, h = by_index[j][0]["bbox_xywh"]
        crops[i] = (max(0.0, x - pad * w), max(0.0, y - pad * h),
                    min(1.0, x + w + pad * w), min(1.0, y + h + pad * h))
    return crops


def extract_images(video_path: Path, indices: list[int], *, cfg, crops: dict | None = None
                   ) -> dict[int, str]:
    """Decode the clip once, encoding only the sampled frames, each cropped to its own box."""
    wanted = set(indices)
    images: dict[int, str] = {}
    cap = cv2.VideoCapture(str(video_path))
    try:
        i = 0
        while len(images) < len(wanted):
            if i not in wanted:
                if not cap.grab():
                    break
            else:
                ok, frame = cap.read()
                if not ok:
                    break
                crop = (crops or {}).get(i, cfg.CROP)
                images[i] = encode_frame(frame, long_edge=cfg.IMAGE_LONG_EDGE,
                                         crop=crop, quality=cfg.JPEG_QUALITY)
            i += 1
    finally:
        cap.release()
    return images


def build_questions(cfg) -> dict:
    from typesafe_sdk import Choice

    return {QUESTION_ID: Choice(instructions=cfg.INSTRUCTIONS, criteria=dict(cfg.LABELS))}


def request_settings(cfg) -> dict:
    """Everything that shapes a read's answer, for the cache key and run.json."""
    return {
        "model": cfg.READ_MODEL,
        "state": cfg.STATE,
        "instructions": cfg.INSTRUCTIONS,
        "labels": cfg.LABELS,
        "sample_fps": cfg.SAMPLE_FPS,
        "image_long_edge": cfg.IMAGE_LONG_EDGE,
        "image_detail": cfg.IMAGE_DETAIL,
        "jpeg_quality": cfg.JPEG_QUALITY,
        "crop": "person" if cfg.CROP_TO_PERSON else cfg.CROP,
        "person_crop_pad": cfg.PERSON_CROP_PAD if cfg.CROP_TO_PERSON else None,
    }


def cache_path(mp4: Path, cache_dir: Path, *, settings: dict, crop_source: str | None = None
               ) -> Path:
    """Keyed on the converted clip, the question *and* where the crops came from,
    so an edited prompt or a re-run pose never reuses a stale answer."""
    stat = mp4.stat()
    key = json.dumps({"sha256": file_sha256(mp4, stat.st_size, stat.st_mtime_ns),
                      "crop_source": crop_source, **settings}, sort_keys=True)
    return cache_dir / f"reads.{hashlib.sha256(key.encode()).hexdigest()[:12]}.json"


def load_cache(path: Path) -> list[Read] | None:
    if not path.is_file():
        return None
    return [Read(**r) for r in json.loads(path.read_text())["reads"]]


def save_cache(path: Path, reads: list[Read], *, stamp: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"created": stamp, "reads": [asdict(r) for r in reads]}))


async def _read_all(images: dict[int, str], fps: float, *, api_key: str, cfg, on_done) -> list[Read]:
    from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy, TypeSafeError

    questions = build_questions(cfg)
    gate = asyncio.Semaphore(cfg.CONCURRENCY)

    async def one(client, index: int, url: str) -> Read:
        async with gate:
            t0 = time.perf_counter()
            try:
                r = await client.system_one(
                    cfg.STATE, questions,
                    extra_body={"content": [
                        {"type": "image_url", "image_url": {"url": url, "detail": cfg.IMAGE_DETAIL}},
                    ]},
                )
            except TypeSafeError as exc:
                on_done()
                # First line only: a gateway 502 is a page of HTML.
                return Read(index, index / fps, None, {}, None,
                            time.perf_counter() - t0, {}, error=str(exc).splitlines()[0][:160])
            latency = time.perf_counter() - t0

        answer = r.answers[QUESTION_ID]
        # The SDK's Usage model drops the gateway's additions (cost, image
        # tokens); the raw body keeps them, so a run can be priced.
        usage = json.loads(r.raw_http_response.text).get("usage", {})
        on_done()
        return Read(index, index / fps, answer.choice, dict(answer.probabilities),
                    answer.confidence, latency, usage)

    async with AsyncTypeSafeClient(
        api_key=api_key, base_url=cfg.TYPESAFE_BASE_URL, model=cfg.READ_MODEL,
        timeout=cfg.READ_TIMEOUT, retry=RetryPolicy(max_retries=cfg.MAX_RETRIES),
    ) as client:
        reads = await asyncio.gather(*(one(client, i, url) for i, url in images.items()))
    return sorted(reads, key=lambda r: r.index)


def read_frames(images: dict[int, str], fps: float, *, api_key: str, cfg, console) -> list[Read]:
    """Ask the question of every sampled frame, *cfg.CONCURRENCY* reads at a time."""
    columns = (
        TextColumn("[cyan]reading[/]"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("{task.completed}/{task.total} frames"),
        TimeElapsedColumn(),
    )
    with Progress(*columns, console=console, transient=True) as progress:
        task = progress.add_task("read", total=len(images))
        return asyncio.run(_read_all(images, fps, api_key=api_key, cfg=cfg,
                                     on_done=lambda: progress.advance(task)))


@dataclass
class Signal:
    """P(positive label) at every output frame, built from the sparse reads."""

    read_times: np.ndarray    # seconds, one per successful read
    raw: np.ndarray           # P(positive) as returned
    smoothed: np.ndarray      # after the median window
    per_frame: np.ndarray     # interpolated onto every video frame
    fps: float

    def at(self, frame: int) -> float:
        return float(self.per_frame[min(frame, len(self.per_frame) - 1)])


def _median(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) < window:
        return values.copy()
    half = window // 2
    padded = np.pad(values, half, mode="edge")
    return np.array([np.median(padded[i:i + window]) for i in range(len(values))])


def build_signal(reads: list[Read], n_frames: int, fps: float, *, cfg) -> Signal:
    ok = [r for r in reads if r.ok]
    if not ok:
        first = next((r.error for r in reads if r.error), "no reads")
        raise RuntimeError(f"every read failed ({first}); the next run retries them")
    t = np.array([r.time for r in ok])
    raw = np.array([r.probabilities[cfg.POSITIVE_LABEL] for r in ok])
    smoothed = _median(raw, cfg.SMOOTH_READS)

    frame_t = np.arange(n_frames) / fps
    if cfg.INTERPOLATE == "hold":
        idx = np.clip(np.searchsorted(t, frame_t, side="right") - 1, 0, len(t) - 1)
        per_frame = smoothed[idx]
    else:
        per_frame = np.interp(frame_t, t, smoothed)
    return Signal(t, raw, smoothed, per_frame, fps)
