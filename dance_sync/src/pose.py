"""The Gateway request and the shape of what comes back."""

from __future__ import annotations

import base64
import hashlib
import itertools
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .skeleton import KPT_NAMES
from .video import file_sha256

CONTENT_OBJECT_KPTS = "vid.pose.kpts"

# Bumped when the request or the response shape this module reads changes, so a
# cache written against the old one is a miss rather than a confusing crash.
CONTRACT = "flat-envelope-v1"


@dataclass
class PoseResult:
    """Frames of poses, keyed by source-frame index."""

    payload: dict
    frames: list[dict]
    by_index: dict[int, list[dict]]
    elapsed: float
    usage: dict | None

    @property
    def track_ids(self) -> list[int]:
        ids = {p.get("track_id") for f in self.frames for p in f["persons"]}
        return sorted(i for i in ids if i is not None)

    @property
    def n_posed(self) -> int:
        return sum(1 for f in self.frames if f["persons"])


def build_request(video: Path, *, every_frame: bool, fps: float, n_frames: int,
                  video_fps: float, video_max_frames, precision: int) -> tuple[str, dict]:
    """Return ``(base64_video, extra_body)``.

    ``video_fps`` is the *detector* cadence, not a sampling rate — pose runs on
    every decoded frame regardless. It reaches stride 1 (detect on every frame)
    as soon as it is >= the decoded rate. ``video_max_frames`` is what actually
    controls how many frames are decoded, posed and billed.
    """
    if every_frame:
        video_fps, video_max_frames = fps, n_frames

    extra_body = {"method": "pose", "video_fps": video_fps, "precision": precision}
    if video_max_frames is not None:
        extra_body["video_max_frames"] = video_max_frames

    return base64.b64encode(video.read_bytes()).decode("ascii"), extra_body


def unwrap(payload: dict) -> tuple[list[dict], dict[int, list[dict]]]:
    """Regroup the flat record list into the per-frame shape this demo works in.

    The gateway returns one flat list of person records, each tagged with the
    `frame_id` it came from, beside a `frames` list naming every frame it
    sampled. Everything downstream of here thinks in frames, so the records are
    grouped back into `{"index": ..., "persons": [...]}` once, here, rather than
    at every call site.

    `frames` is the authority on which frames exist, not the records: nobody is
    detected in some of them, those carry no records at all, and rebuilding the
    clip from the records alone would silently close the gap instead of showing
    it.

    `by_index` starts out holding each frame's own `persons` list, but that is
    only true until something filters them: every cleaning pass here rebinds
    `frame["persons"]` rather than editing the list in place, so a caller that
    cleans must rebuild `by_index` afterwards, as `main.py` does.

    The joint order is checked rather than assumed. `kpts_labels` names the
    joints the model actually returned, and every index in `skeleton.py` — which
    wrist, which ankle — is a position in that list. A model that returned a
    different layout would still draw, silently, as a tangle of limbs.
    """
    content = payload.get("content")
    if not isinstance(content, dict):
        raise RuntimeError(f"Unexpected response envelope: {json.dumps(payload)[:400]}")

    got = content.get("object")
    if got != CONTENT_OBJECT_KPTS:
        raise RuntimeError(
            f"Expected a video pose payload ({CONTENT_OBJECT_KPTS}), got {got!r}. "
            "Did the request send an image_url instead of a video_url?"
        )

    labels = content.get("kpts_labels")
    if labels is not None and list(labels) != KPT_NAMES:
        raise RuntimeError(
            f"The model returned {len(labels)} joints in an order this demo does "
            f"not know: {list(labels)}. src/skeleton.py is written against COCO-17."
        )

    by_index: dict[int, list[dict]] = {
        int(f["frame_id"]): [] for f in content.get("frames") or []}
    for person in content.get("items") or []:
        index = person.get("frame_id")
        if index is not None:
            by_index.setdefault(int(index), []).append(person)

    frames = [{"index": i, "persons": by_index[i]} for i in sorted(by_index)]
    return frames, by_index


def request_poses(client, *, model: str, video_b64: str, extra_body: dict) -> dict:
    """One blocking chat-completions call, returning the parsed JSON payload."""
    response = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": [{
                "type": "video_url",
                "video_url": {"url": f"data:video/mp4;base64,{video_b64}"},
            }],
        }],
        response_format={"type": "json_object"},
        extra_body=extra_body,
    )
    payload = json.loads(response.choices[0].message.content)
    usage = response.usage.model_dump() if response.usage else None
    return payload, usage


def cache_path(video: Path, cache_dir: Path, *, model: str, extra_body: dict) -> Path:
    """Where this exact request's poses live.

    Keyed on the converted video's identity *and* every field that shapes the
    result, so a cache hit can only mean "same clip, same model, same request".
    Changing `video_fps`, the model, or the converted video's bytes all produce
    a different path rather than silently reusing stale poses.
    """
    stat = video.stat()
    key = json.dumps(
        {
            "sha256": file_sha256(video, stat.st_size, stat.st_mtime_ns),
            "model": model,
            "extra_body": extra_body,
            "contract": CONTRACT,
        },
        sort_keys=True,
    )
    digest = hashlib.sha256(key.encode()).hexdigest()[:12]
    return cache_dir / f"poses.{digest}.json"


def save_cache(path: Path, payload: dict, usage: dict | None, timing_dict: dict,
               *, stamp: str) -> None:
    """Store the response plus the timings it was actually measured with."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "created": stamp,
        "payload": payload,
        "usage": usage,
        "timing": timing_dict,
    }))


def load_cache(path: Path) -> tuple[dict, dict | None, dict, str] | None:
    """Read a cached response, or None if it is missing or unreadable."""
    if not path.is_file():
        return None
    try:
        blob = json.loads(path.read_text())
        return blob["payload"], blob.get("usage"), blob.get("timing") or {}, blob.get(
            "created", "unknown")
    except (json.JSONDecodeError, KeyError, OSError):
        return None


def _area(box) -> float:
    return float(box[2] * box[3])


def _iou(a, b) -> float:
    """Intersection over union for two normalized ``[x, y, w, h]`` boxes."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x0, y0 = max(ax, bx), max(ay, by)
    x1, y1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = aw * ah + bw * bh - inter
    return float(inter / union) if union > 0 else 0.0


def _degenerate(box) -> bool:
    """A box with non-positive width or height. There are 27 in this clip."""
    return box[2] <= 0 or box[3] <= 0


@dataclass
class CleanReport:
    """What the cleanup pass removed, so nothing is dropped silently."""

    before: int = 0
    after: int = 0
    blank: int = 0
    degenerate: int = 0
    same_track: int = 0
    tiny: int = 0
    frozen: int = 0
    ghost_tracks: int = 0
    merged: int = 0
    merges: list[tuple] = field(default_factory=list)   # (kept_id, absorbed_id, distance)
    duplicates: int = 0
    capped: int = 0
    tracks: list[dict] = field(default_factory=list)   # per-track stats + verdict

    @property
    def kept_track_ids(self) -> list[int]:
        return [t["track_id"] for t in self.tracks if t["kept"]]

    @property
    def dropped(self) -> int:
        return self.before - self.after


def clean(frames: list[dict], *, blank_indices: set[int], min_box_area: float,
          drop_frozen: bool, min_track_fraction: float, min_track_area: float,
          max_persons: int | None, iou: float,
          merge_distance: float = 0.0, merge_min_frames: int = 5) -> CleanReport:
    """Reduce the raw detections to one box per real subject, in place.

    The model reports ten track ids on a clip with three dancers. Seven distinct
    problems produce those extra seven, and each needs its own pass:

    1. **Poses on a blank frame** — the clip's last frame is fully black, and the
       model returns three poses for it: the detector finds nothing, the tracker
       predicts each box forward a step, and ViTPose runs on a crop of pure
       black. Every one of those poses is invented, so the frame is rejected
       whole rather than per subject. Which frames are blank is a question about
       pixels, so *src.video.blank_frames* answers it and passes the indices in.
    2. **Degenerate boxes** — non-positive width or height. Not a subject at all.
    3. **One track, two boxes in a frame** — a track id is an identity, so a
       frame listing the same id twice is a duplicate by definition, whatever
       the boxes look like. This is exact where geometry is a guess, so it runs
       first and catches 114 boxes here.
    4. **Boxes too small to be a person** — the two dancers who enter late each
       have a real track id whose *early* frames sit on background clutter, so a
       per-track verdict cannot reject them; the individual box has to go. This
       is an absolute floor, not a fraction of the track's own size: the centre
       dancer starts far from the camera at area 0.068 and walks in to 0.257, so
       "much smaller than usual for this track" is a description of that dancer,
       not of a ghost. The two populations are far apart — real detections start
       at 0.068, ghost boxes stop at 0.03 — so anything in that gap gives the
       same answer.
    5. **Frozen boxes** — a box byte-identical to the same track's box on the
       previous frame. That is a tracker coasting on a target it has lost, not a
       subject holding perfectly still: a real body's box changes every frame
       from pose noise alone, and across this clip's three dancers there is not
       one frozen pair (0 of 1653) against a 31-frame frozen run on the ghost.
    6. **Ghost tracks** — a box tracked on a background object, a shadow, or a
       dancer's own raised arm, for its whole life. Two properties separate them
       from people with wide margins on this clip, so both are required rather
       than either:

           track          lifetime   median area
           dancers        0.86-1.00  0.104-0.257
           ghosts        <=0.29     <=0.054

       Lifetime is the stronger signal — a dancer is on screen for seconds, a
       false positive flickers — and area rules out the arm ghosts that happen
       to persist. Note what is *not* used: aspect ratio looks tempting (ghost
       boxes are slivers) but a dancer mid-kick reaches w/h 1.2 while a ghost
       drops to 0.39, so the ranges overlap and it would cost real detections.
    7. **Near-identical boxes across two ids** — the tracker hands identity over
       by running the old and new track together for a few frames before the old
       one ages out, so one body briefly reports twice at IoU ~0.97.

    Deliberately absent: suppressing a small box that sits *inside* a larger one.
    It reads as the obvious fix for arm ghosts, and it costs 183 real frames of
    the right-hand dancer, whose box is often mostly covered by the center
    dancer's much larger one.

    Each surviving person also gets a ``slot``: its subject's rank in order of
    first appearance, 0, 1, 2. That is what the overlay colors by — track ids
    are arbitrary (this clip keeps 0, 1 and 4), and indexing a palette with them
    handed two of three dancers nearly the same blue.
    """
    report = CleanReport(before=sum(len(f.get("persons") or []) for f in frames))

    # 1 — no image content means no pose could have been measured.
    if blank_indices:
        for frame in frames:
            if frame["index"] in blank_indices and frame.get("persons"):
                report.blank += len(frame["persons"])
                frame["persons"] = []

    # 2 + 3 — per-frame, and independent of every other subject in the frame.
    for frame in frames:
        persons = frame.get("persons") or []

        kept = [p for p in persons if not _degenerate(p["bbox_xywh"])]
        report.degenerate += len(persons) - len(kept)

        best: dict = {}
        anonymous: list[dict] = []
        for person in kept:
            tid = person.get("track_id")
            if tid is None:
                anonymous.append(person)   # untracked: nothing to collapse on
            elif tid not in best or _area(person["bbox_xywh"]) > _area(best[tid]["bbox_xywh"]):
                best[tid] = person
        collapsed = list(best.values()) + anonymous
        report.same_track += len(kept) - len(collapsed)

        frame["persons"] = collapsed

    # 4 — an absolute floor; see the docstring for why it is not relative.
    for frame in frames:
        kept = [p for p in frame["persons"] if _area(p["bbox_xywh"]) >= min_box_area]
        report.tiny += len(frame["persons"]) - len(kept)
        frame["persons"] = kept

    # 5 — needs the previous frame's box for the *same* track, so it walks the
    # clip in order rather than treating each frame on its own.
    if drop_frozen:
        previous: dict = {}
        for frame in frames:
            kept = []
            for person in frame["persons"]:
                box = tuple(person["bbox_xywh"])
                tid = person.get("track_id")
                if tid is not None and previous.get(tid) == box:
                    report.frozen += 1
                else:
                    kept.append(person)
                if tid is not None:
                    previous[tid] = box
            frame["persons"] = kept

    # 5b — one dancer wearing two ids. This has to run *before* the lifetime
    # test below, because that test is what would kill the shorter half.
    #
    # The v2 source splits the left-hand dancer at frame 386: id 1 runs 88-428,
    # id 5 runs 386-545, and they share 43 frames. Judged alone id 5 has a 0.28
    # lifetime, which lands it in the ghost band and cost 91 scored frames.
    #
    # The test is **keypoint agreement, not box overlap**. Box IoU is what the
    # per-frame handoff pass uses and it does not separate this case: the two
    # boxes on the same body have a median IoU of 0.49, against 0.22 for a ghost
    # sitting on a real dancer — overlapping ranges. The keypoints are decisive,
    # because two ids on one body are reading the same pose:
    #
    #     pair    median keypoint distance
    #     (1, 5)  0.0062   <- the same dancer, twice
    #     (2, 3)  0.0292      an arm ghost on a real dancer
    #     others  0.15-0.79   different dancers
    #
    # Both tracks must also be person-sized, which is what keeps the arm ghost
    # out: area is the reliable ghost discriminator, and it is only *lifetime*
    # this pass exists to rescue.
    if merge_distance > 0:
        by_id: dict = {}
        for frame in frames:
            for person in frame["persons"]:
                by_id.setdefault(person.get("track_id"), []).append(
                    (frame["index"], person))

        areas = {tid: sorted(_area(p["bbox_xywh"]) for _, p in rows)[len(rows) // 2]
                 for tid, rows in by_id.items()}
        kpts = {tid: {i: np.asarray(p["kpts_xy"], float) for i, p in rows}
                for tid, rows in by_id.items()}

        parent = {tid: tid for tid in by_id}

        def root(tid):
            while parent[tid] != tid:
                tid = parent[tid]
            return tid

        ids = sorted(by_id, key=lambda t: (t is None, t))
        for a, b in itertools.combinations(ids, 2):
            if areas[a] < min_track_area or areas[b] < min_track_area:
                continue
            shared = kpts[a].keys() & kpts[b].keys()
            if len(shared) < merge_min_frames:
                continue
            distances = sorted(
                float(np.mean(np.linalg.norm(kpts[a][i] - kpts[b][i], axis=1)))
                for i in shared)
            median = distances[len(distances) // 2]
            if median > merge_distance:
                continue
            ra, rb = root(a), root(b)
            if ra == rb:
                continue
            # The earlier-appearing id wins, so slot order stays first-appearance.
            first = {t: min(i for i, _ in by_id[t]) for t in (ra, rb)}
            keep, absorb = sorted((ra, rb), key=lambda t: first[t])
            parent[absorb] = keep
            report.merges.append((keep, absorb, round(median, 4)))

        if report.merges:
            for frame in frames:
                for person in frame["persons"]:
                    person["track_id"] = root(person.get("track_id"))
                # The merge can leave two boxes under one id in the overlap
                # frames; keep the larger, exactly as pass 3 does for a repeated
                # id, since one id is one identity.
                best: dict = {}
                for person in frame["persons"]:
                    tid = person["track_id"]
                    if tid not in best or _area(person["bbox_xywh"]) > _area(best[tid]["bbox_xywh"]):
                        best[tid] = person
                report.merged += len(frame["persons"]) - len(best)
                frame["persons"] = list(best.values())

    # 6 — a track's verdict needs the whole clip, so this pass is global.
    n_frames = len(frames) or 1
    seen: dict = {}
    for frame in frames:
        for person in frame["persons"]:
            seen.setdefault(person.get("track_id"), []).append(_area(person["bbox_xywh"]))

    keep_ids = set()
    for tid, areas in sorted(seen.items(), key=lambda kv: (kv[0] is None, kv[0])):
        fraction = len(areas) / n_frames
        median = sorted(areas)[len(areas) // 2]
        kept = fraction >= min_track_fraction and median >= min_track_area
        if kept:
            keep_ids.add(tid)
        report.tracks.append({
            "track_id": tid, "frames": len(areas), "lifetime": round(fraction, 3),
            "median_area": round(median, 4), "kept": kept,
            "reason": None if kept else
            ("short-lived" if fraction < min_track_fraction else "too small"),
        })

    for frame in frames:
        kept = [p for p in frame["persons"] if p.get("track_id") in keep_ids]
        report.ghost_tracks += len(frame["persons"]) - len(kept)
        frame["persons"] = kept

    # 7 — largest first, so a handoff keeps the better-framed box of the pair.
    for frame in frames:
        kept: list[dict] = []
        for person in sorted(frame["persons"], key=lambda p: -_area(p["bbox_xywh"])):
            if all(_iou(person["bbox_xywh"], k["bbox_xywh"]) < iou for k in kept):
                kept.append(person)
        report.duplicates += len(frame["persons"]) - len(kept)

        if max_persons and len(kept) > max_persons:
            report.capped += len(kept) - max_persons
            kept = kept[:max_persons]

        frame["persons"] = kept

    # Palette slots, in order of first appearance among the survivors.
    slots: dict = {}
    for frame in frames:
        for person in frame["persons"]:
            tid = person.get("track_id")
            person["slot"] = slots.setdefault(tid, len(slots))
    for row in report.tracks:
        row["slot"] = slots.get(row["track_id"])

    report.after = sum(len(f["persons"]) for f in frames)
    return report
