"""The barbell plate: SAM 3.1 tracking it through the clip, one outline per sample.

SAM 3.1's ``track`` method takes the whole video in one call and returns every
instance of the prompt in every sampled frame, each with a ``track_id``. On a
deadlift the id is not to be trusted: SAM drops the plate as it leaves the floor
and picks it back up under a new id, so a clip of three reps comes back as three
or four tracks of one plate, plus the odd small false positive. None of that
matters here, because there is exactly one plate in shot. Each sampled frame
keeps the one detection that is plate-sized and nearest to where the plate just
was, whatever id SAM gave it.

The plate is a rigid disc of known size, which the rest of the demo leans on
twice: between samples its outline is the nearest sampled one *translated*, and
its height in pixels is a ruler (an Olympic plate is 45 cm, `PLATE_DIAMETER_CM`),
so bar height can be quoted in centimeters rather than in fractions of the frame.
"""

from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .video import file_sha256

CONTENT_OBJECT = "vid.segment.masks"

# Bumped when the request or the response shape this module reads changes.
CONTRACT = "sam31-plate-track-v1"

# SAM 3.1's tracker holds one frame of memory per sample and the gateway caps a
# request at this many; a longer clip is cut into segments instead.
TRACK_MAX_FRAMES = 128


def request_track(client, *, model: str, video_b64: str, prompt: str,
                  skip_frames: int, max_frames: int) -> tuple[dict, dict | None]:
    """One call: every instance of *prompt*, tracked through the clip."""
    response = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{video_b64}"}},
            ],
        }],
        response_format={"type": "json_object"},
        extra_body={"method": "track", "method_params": {
            "prompt": prompt, "video_skip_frames": skip_frames,
            "video_max_frames": max_frames, "mask_format": "png", "polygons": True}},
    )
    payload = json.loads(response.choices[0].message.content)
    usage = response.usage.model_dump() if response.usage else None
    return payload, usage


def unwrap(payload: dict) -> tuple[list[dict], list[dict]]:
    """``(items, frames)`` out of the envelope, with the payload tag checked."""
    content = payload.get("content")
    if not isinstance(content, dict):
        raise RuntimeError(f"Unexpected response envelope: {json.dumps(payload)[:400]}")
    got = content.get("object")
    if got != CONTENT_OBJECT:
        raise RuntimeError(f"Expected a {CONTENT_OBJECT} payload, got {got!r}.")
    return content.get("items") or [], content.get("frames") or []


# ── long clips: tracking in segments ─────────────────────────────────────────

def segment_plan(n_frames: int, stride: int, *, max_frames: int = TRACK_MAX_FRAMES
                 ) -> list[tuple[int, int]]:
    """``[(start, length), ...]`` covering the clip at *stride* within the per-call cap."""
    span = max_frames * stride
    if n_frames <= span:
        return [(0, n_frames)]
    count = -(-n_frames // span)
    length = -(-n_frames // count)
    return [(i * length, min(length, n_frames - i * length)) for i in range(count)]


def shift(payload: dict, *, frame_offset: int, track_offset: int) -> dict:
    """Renumber one segment's reply into the whole clip's frame and track space."""
    items, frames = unwrap(payload)
    for item in items:
        if item.get("frame_id") is not None:
            item["frame_id"] = int(item["frame_id"]) + frame_offset
        if item.get("track_id") is not None:
            # The frame's label map still paints SAM's own id, so keep it for the lookup.
            item.setdefault("mask_id", int(item["track_id"]))
            item["track_id"] = int(item["track_id"]) + track_offset
    for frame in frames:
        frame["frame_id"] = int(frame["frame_id"]) + frame_offset
    return payload


def concat(payloads: list[dict]) -> dict:
    """Fuse already-shifted segment replies into one, as if a single call made it."""
    items: list[dict] = []
    frames: list[dict] = []
    for payload in payloads:
        part_items, part_frames = unwrap(payload)
        items += part_items
        frames += part_frames
    head = dict(payloads[0])
    head["content"] = {"object": CONTENT_OBJECT, "items": items,
                       "frames": sorted(frames, key=lambda f: f["frame_id"])}
    return head


# ── cache ────────────────────────────────────────────────────────────────────

def cache_path(video: Path, cache_dir: Path, *, model: str, prompt: str, settings: dict) -> Path:
    stat = video.stat()
    key = json.dumps({
        "sha256": file_sha256(video, stat.st_size, stat.st_mtime_ns),
        "model": model, "prompt": prompt, "settings": settings, "contract": CONTRACT,
    }, sort_keys=True)
    return cache_dir / f"plate.{hashlib.sha256(key.encode()).hexdigest()[:12]}.json"


def save_cache(path: Path, payload: dict, usage: dict | None, *, stamp: str) -> None:
    """The raw reply, so every selection threshold below stays free to re-tune."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"created": stamp, "payload": payload, "usage": usage}))


def load_cache(path: Path):
    if not path.is_file():
        return None
    try:
        blob = json.loads(path.read_text())
        return blob["payload"], blob.get("usage"), blob.get("created", "unknown")
    except (json.JSONDecodeError, KeyError, OSError):
        return None


# ── the plate, per frame ─────────────────────────────────────────────────────

def _decode_label_map(mask: dict | None) -> np.ndarray | None:
    """A frame's label map as uint8: pixel = ``track_id``, 0 = no object."""
    if not isinstance(mask, dict) or mask.get("format") != "png":
        return None
    data = mask.get("data")
    if not isinstance(data, str):
        return None
    raw = base64.b64decode(data.split(",", 1)[-1])
    labels = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if labels is None:
        return None
    if labels.ndim == 3:
        labels = labels[:, :, 0]
    return labels.astype(np.uint8)


def _contour(mask: np.ndarray) -> np.ndarray:
    """The largest external contour of a boolean mask, in that mask's pixels."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.zeros((0, 2))
    return max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)


@dataclass
class PlateTrack:
    """The plate at every sampled frame, and interpolated onto every video frame.

    Positions are normalized frame coordinates; ``outlines`` are normalized too,
    so one track serves any render size.
    """

    sampled: np.ndarray            # frame indices SAM looked at
    found: np.ndarray              # bool per sample: a plate was kept
    cx: np.ndarray                 # center, per sample (nan where not found)
    cy: np.ndarray
    height: np.ndarray             # bbox height, per sample
    outlines: dict[int, np.ndarray] = field(default_factory=dict)   # sample frame -> (N, 2)
    track_ids: list[int] = field(default_factory=list)
    per_frame_cx: np.ndarray = field(default_factory=lambda: np.zeros(0))
    per_frame_cy: np.ndarray = field(default_factory=lambda: np.zeros(0))

    @property
    def n_found(self) -> int:
        return int(self.found.sum())

    @property
    def median_height(self) -> float:
        """Plate diameter as a fraction of frame height: the ruler."""
        return float(np.nanmedian(self.height[self.found])) if self.found.any() else float("nan")

    def outline_at(self, frame: int) -> np.ndarray | None:
        """The nearest sampled outline, moved to where the plate is at *frame*.

        The plate is rigid and barely rotates in shot, so a translated outline is
        the plate, rather than an approximation that happens to look like it.
        """
        if not self.outlines:
            return None
        keys = np.fromiter(self.outlines.keys(), int)
        nearest = int(keys[np.argmin(np.abs(keys - frame))])
        i = int(np.where(self.sampled == nearest)[0][0])
        dx = self.per_frame_cx[frame] - self.cx[i]
        dy = self.per_frame_cy[frame] - self.cy[i]
        return self.outlines[nearest] + [dx, dy]


def select(payload: dict, n_frames: int, *, min_score: float, min_area_fraction: float,
           size_tolerance: float) -> PlateTrack:
    """One plate per sampled frame, chosen by size and continuity, not by track id."""
    items, frames = unwrap(payload)
    sampled = np.array(sorted(int(f["frame_id"]) for f in frames), dtype=int)
    masks = {int(f["frame_id"]): f.get("mask") for f in frames}

    by_frame: dict[int, list[dict]] = {}
    for item in items:
        if item.get("frame_id") is None or float(item.get("score") or 0) < min_score:
            continue
        box = item.get("bbox_xywh")
        if isinstance(box, (list, tuple)) and len(box) == 4:
            by_frame.setdefault(int(item["frame_id"]), []).append(item)

    # "Plate-sized" is relative to the plate itself: the largest detection in
    # each frame is nearly always the plate, so their median is its area, and a
    # small false positive on the collar or a shoe falls well short of it.
    largest = [max(v, key=lambda i: float(i.get("area") or 0)) for v in by_frame.values()]
    typical = float(np.median([float(i.get("area") or 0) for i in largest])) if largest else 0.0
    # A plate is a rigid disc, so its height in shot barely changes. A box far
    # off that is SAM merging the plate with the bar or the lifter's shins, and
    # its center is not the plate's: one such box reads as a 45 cm dip.
    typical_h = float(np.median([i["bbox_xywh"][3] for i in largest])) if largest else 0.0

    cx = np.full(len(sampled), np.nan)
    cy = np.full(len(sampled), np.nan)
    height = np.full(len(sampled), np.nan)
    outlines: dict[int, np.ndarray] = {}
    track_ids: set[int] = set()
    previous = None

    for k, frame in enumerate(sampled):
        # No plate-sized box in the clip: skip rather than divide by a zero height.
        if typical_h <= 0:
            continue
        candidates = [i for i in by_frame.get(frame, [])
                      if float(i.get("area") or 0) >= min_area_fraction * typical
                      and abs(i["bbox_xywh"][3] / typical_h - 1) <= size_tolerance]
        if not candidates:
            continue

        def center(item):
            x, y, w, h = item["bbox_xywh"]
            return np.array([x + w / 2, y + h / 2])

        if previous is None:
            best = max(candidates, key=lambda i: float(i.get("area") or 0))
        else:
            best = min(candidates, key=lambda i: float(np.linalg.norm(center(i) - previous)))
        c = center(best)
        cx[k], cy[k] = c
        height[k] = best["bbox_xywh"][3]
        previous = c
        if best.get("track_id") is not None:
            track_ids.add(int(best["track_id"]))

        labels = _decode_label_map(masks.get(frame))
        mask_id = best.get("mask_id", best.get("track_id"))
        if labels is not None and mask_id is not None:
            blob = labels == int(mask_id)
            if blob.any():
                pts = _contour(blob)
                if len(pts) >= 3:
                    outlines[int(frame)] = pts / [labels.shape[1], labels.shape[0]]
                    continue
        rings = best.get("polys_xy") or []
        if rings:
            ring = np.asarray(max(rings, key=len), dtype=np.float64)
            if len(ring) >= 3:
                outlines[int(frame)] = ring

    found = np.isfinite(cy)
    track = PlateTrack(sampled, found, cx, cy, height, outlines, sorted(track_ids))
    if found.any():
        idx = np.arange(n_frames)
        track.per_frame_cx = np.interp(idx, sampled[found], cx[found])
        track.per_frame_cy = np.interp(idx, sampled[found], cy[found])
    else:
        track.per_frame_cx = np.full(n_frames, np.nan)
        track.per_frame_cy = np.full(n_frames, np.nan)
    return track
