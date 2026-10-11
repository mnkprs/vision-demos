"""Poses for a caller outside main.py: this demo's own request, cache and clean-up, in one call.

`main.py` builds the request, finds or pays for the poses and de-duplicates them in steps of its own,
with its console between them. A program that wants only the poses (a few frames to check a clip, or a
long video at a low frame rate) calls `probe` instead of reaching into `pose`, and gets them in one
shape whichever demo answers: every demo carries this same file, and it reads that demo's `pose`
module, whose `unwrap` returns frames in most demos and flat person records with the sampled frame ids
in rock_climbing.

`probe` takes values, not `config`: the caller names the model, the gateway and the request knobs
(`VIDEO_FPS`, `VIDEO_MAX_FRAMES`, `PRECISION` and the duplicate-box settings in this demo's config).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from . import pose


class NotCached(LookupError):
    """A recorded request whose poses are not in the cache (`probe(..., recorded=...)`)."""

    def __init__(self, path: Path):
        super().__init__(f"no cached poses at {path}")
        self.path = path


@dataclass
class Probe:
    """One request's poses, every sampled frame in order."""

    frames: list[dict]      # [{"index", "detected", "persons"}]; `detected` counts before de-duplication
    deduped: bool           # whether `persons` went through this demo's `dedupe`
    extra_body: dict        # the request: as sent, or as recorded
    payload: dict           # the gateway's reply
    cost_usd: float         # 0.0 for poses read from the cache
    cached: bool
    seconds: float | None   # the gateway call's wall time; None when cached


def probe(video: Path, *, model: str, fps: float | None = None, n_frames: int | None = None,
          video_fps: float | None = None, video_max_frames=None, precision: int | None = None,
          every_frame: bool = True, base_url: str | None = None, api_key: str | None = None,
          timeout: float = 60.0, cache_dir: Path | None = None, recorded: dict | None = None,
          max_persons: int | None = None, iou: float | None = None) -> Probe:
    """Pose *video* as main.py does a whole clip, and return the frames.

    A new request is `pose.build_request` on *video* with these knobs (``every_frame`` sends every
    decoded frame: ``fps`` and ``n_frames`` as the video has them). With ``cache_dir`` a request this
    demo already paid for is read back for free, and a new one is saved there; without it the gateway
    is always called. The gateway client is made here, once: ``base_url``, ``api_key``, ``timeout``,
    and no retries, so a failure costs one call and reaches the caller.

    ``recorded`` is a finished run's ``extra_body``: its poses are read from ``cache_dir`` only, never
    requested, and `NotCached` is raised when they are not there.

    ``iou`` (this demo's ``PERSON_IOU_THRESHOLD``) turns on de-duplication, as main.py does it, in a
    demo that has ``dedupe``, capped at ``max_persons``; ``deduped`` says whether it ran.
    """
    if recorded is not None:
        if cache_dir is None:
            raise ValueError("a recorded request is read from cache_dir, and none was given")
        path = pose.cache_path(video, cache_dir, model=model, extra_body=recorded)
        loaded = pose.load_cache(path)
        if loaded is None:
            raise NotCached(path)
        return _result(loaded[0], recorded, cost_usd=0.0, cached=True, seconds=None,
                       max_persons=max_persons, iou=iou)

    video_b64, extra_body = pose.build_request(
        video, every_frame=every_frame, fps=fps, n_frames=n_frames, video_fps=video_fps,
        video_max_frames=video_max_frames, precision=precision)
    path = None if cache_dir is None else pose.cache_path(video, cache_dir, model=model,
                                                          extra_body=extra_body)
    loaded = None if path is None else pose.load_cache(path)
    if loaded is not None:                    # paid for by the request that made it
        return _result(loaded[0], extra_body, cost_usd=0.0, cached=True, seconds=None,
                       max_persons=max_persons, iou=iou)

    if api_key is None:
        raise ValueError("a gateway request needs api_key")
    from openai import OpenAI
    client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout, max_retries=0)
    t0 = time.perf_counter()
    payload, usage = pose.request_poses(client, model=model, video_b64=video_b64,
                                        extra_body=extra_body)
    seconds = time.perf_counter() - t0
    if path is not None:
        pose.save_cache(path, payload, usage, {}, stamp=time.strftime("%Y-%m-%d %H:%M:%S"))
    return _result(payload, extra_body, cost_usd=float((usage or {}).get("cost") or 0),
                   cached=False, seconds=seconds, max_persons=max_persons, iou=iou)


def _result(payload: dict, extra_body: dict, *, cost_usd: float, cached: bool,
            seconds: float | None, max_persons: int | None, iou: float | None) -> Probe:
    frames = _frames(pose.unwrap(payload))
    detected = {f["index"]: len(f["persons"]) for f in frames}
    deduped = iou is not None and hasattr(pose, "dedupe")
    if deduped:
        pose.dedupe(frames, max_persons=max_persons, iou=iou)
    frames = [{"index": f["index"], "detected": detected[f["index"]], "persons": f["persons"]}
              for f in frames]
    return Probe(frames, deduped, extra_body, payload, cost_usd, cached, seconds)


def _frames(unwrapped: tuple) -> list[dict]:
    """`pose.unwrap` as frames, `[{"index", "persons"}]`: frames already (most demos), or
    rock_climbing's flat records and sampled frame ids, grouped by each record's `frame_id`."""
    first, second = unwrapped
    if not isinstance(second, list):
        return first
    by_index: dict[int, list[dict]] = {int(i): [] for i in second}
    for person in first:
        if person.get("frame_id") is not None:
            by_index.setdefault(int(person["frame_id"]), []).append(person)
    return [{"index": i, "persons": by_index[i]} for i in sorted(by_index)]
