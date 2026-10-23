"""Probing and (cached) conversion of the source clip."""

from __future__ import annotations

import functools
import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn


@dataclass(frozen=True)
class VideoInfo:
    """Geometry of a decoded clip, as OpenCV will see it."""

    path: Path
    width: int
    height: int
    fps: float
    n_frames: int

    @property
    def duration(self) -> float:
        return self.n_frames / self.fps if self.fps else 0.0

    def __str__(self) -> str:
        return (
            f"{self.width}x{self.height} @ {self.fps:.2f} fps — "
            f"{self.n_frames} frames, {self.duration:.1f}s"
        )


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed:\n{proc.stderr[-3000:]}")
    return proc


def probe_source(path: Path) -> dict:
    """ffprobe the source, including the rotation OpenCV would miss."""
    proc = _run([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,width,height,nb_frames,duration,pix_fmt",
        "-show_entries", "stream_side_data=rotation",
        "-of", "json", str(path),
    ])
    stream = json.loads(proc.stdout)["streams"][0]
    rotation = 0
    for side in stream.get("side_data_list") or []:
        if "rotation" in side:
            rotation = int(side["rotation"])
    return {
        "codec": stream.get("codec_name"),
        "width": int(stream.get("width", 0)),
        "height": int(stream.get("height", 0)),
        "pix_fmt": stream.get("pix_fmt"),
        "n_frames": int(stream.get("nb_frames") or 0),
        "duration": float(stream.get("duration") or 0.0),
        "rotation": rotation,
    }


def probe_audio(path: Path) -> dict | None:
    """The source's first audio stream, or None if it has none.

    Asked separately from :func:`probe_source` because the answer decides
    whether the final mux can happen at all — a clip with no audio track is not
    an error, it just means the overlay stays silent.
    """
    proc = _run([
        "ffprobe", "-v", "error", "-select_streams", "a:0",
        "-show_entries", "stream=codec_name,channels,sample_rate,bit_rate,duration",
        "-of", "json", str(path),
    ])
    streams = json.loads(proc.stdout).get("streams") or []
    if not streams:
        return None
    stream = streams[0]
    return {
        "codec": stream.get("codec_name"),
        "channels": int(stream.get("channels") or 0),
        "sample_rate": int(stream.get("sample_rate") or 0),
        "bit_rate": int(stream.get("bit_rate") or 0) or None,
        "duration": float(stream.get("duration") or 0.0),
    }


@functools.lru_cache(maxsize=None)
def file_sha256(path: Path, size: int, mtime_ns: int) -> str:
    """sha256 of *path*, read in chunks and once per run (*size* and *mtime_ns* spot a rewrite)."""
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def cache_path(src: Path, cache_dir: Path, *, target_height, trim_seconds, crf) -> Path:
    """Deterministic name for the converted MP4.

    Keyed on the source's identity *and* the settings that shaped the output, so
    changing MAX_LONG_EDGE produces a different file rather than silently
    reusing the previous one.
    """
    stat = src.stat()
    key = json.dumps(
        {
            # The bytes, not the path or mtime: a copy of the clip elsewhere reuses this.
            "sha256": file_sha256(src, stat.st_size, stat.st_mtime_ns),
            "target_height": target_height,
            "trim_seconds": trim_seconds,
            "crf": crf,
        },
        sort_keys=True,
    )
    digest = hashlib.sha256(key.encode()).hexdigest()[:12]
    # Not the source's name either: the same bytes under another name reuse it.
    return cache_dir / f"clip.{digest}.mp4"


def convert(
    src: Path,
    dst: Path,
    *,
    target_height: int | None,
    trim_seconds: float | None,
    crf: int,
    total_frames: int,
    console,
) -> None:
    """Transcode to H.264 MP4, printing ffmpeg's own frame counter as progress."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".partial.mp4")

    cmd = ["ffmpeg", "-y", "-nostats", "-loglevel", "error", "-progress", "pipe:1", "-i", str(src)]
    if trim_seconds:
        cmd += ["-t", str(trim_seconds)]
    if target_height:
        # Height is set; width follows the aspect ratio. `-2` keeps it even (H.264
        # needs that) and `min(...,ih)` means this only ever downscales — asking
        # for 1080 from a 720p source leaves it at 720 rather than upsampling
        # pixels that were never there.
        #
        # `ih` here is the *rotated* height: ffmpeg applies the container's
        # rotation metadata before filters run, which is why a phone clip stored
        # 2932x1650 scales as the 1650x2932 portrait it actually is.
        cmd += ["-vf", f"scale=-2:'min({target_height},ih)'"]
    cmd += [
        "-c:v", "libx264", "-preset", "fast", "-crf", str(crf),
        "-pix_fmt", "yuv420p",   # 10-bit HEVC -> 8-bit H.264, which everything decodes
        "-an",                   # audio is dead weight in the upload
        str(tmp),
    ]

    columns = [
        TextColumn("[cyan]converting[/]"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("{task.completed}/{task.total} frames"),
        TimeElapsedColumn(),
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        with Progress(*columns, console=console, transient=True) as progress:
            task = progress.add_task("convert", total=total_frames or None)
            for line in proc.stdout:
                key, _, value = line.strip().partition("=")
                if key == "frame" and value.isdigit():
                    progress.update(task, completed=min(int(value), total_frames or int(value)))
            proc.wait()
    finally:
        if proc.poll() is None:
            proc.kill()

    if proc.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg conversion failed:\n{proc.stderr.read()[-3000:]}")

    tmp.replace(dst)   # atomic: a killed run never leaves a half-file in the cache


def inspect(path: Path) -> VideoInfo:
    """Read geometry the way the renderer will, i.e. through OpenCV."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open {path}")
    info = VideoInfo(
        path=path,
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        fps=cap.get(cv2.CAP_PROP_FPS) or 30.0,
        n_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    )
    cap.release()
    return info


def blank_frames(path: Path, *, max_luma: int) -> list[int]:
    """Indices of frames with no image content at all.

    This clip's last frame is fully black — it is in the source `.MOV`, not an
    artifact of conversion — and the model still returns three poses for it: the
    detector finds nothing, so the tracker predicts each box forward one step
    and ViTPose runs on a crop of pure black. Nothing downstream can tell that
    from a real pose, so the frames are identified here, from the pixels.

    The test is the *brightest* pixel, not the mean. A night clip is mostly dark
    and a mean-based threshold would be a judgement call about how dark is too
    dark; a frame whose brightest pixel is near zero has no content by
    definition. On this clip the separation is total — frame 546's maximum is 0
    and all 546 others are 255.
    """
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open {path}")

    blank, index = [], 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).max() <= max_luma:
                blank.append(index)
            index += 1
    finally:
        cap.release()
    return blank


def encode_h264(src: Path, dst: Path, *, crf: int, audio_from: Path | None = None,
                audio_codec: str = "copy", trim_seconds: float | None = None) -> None:
    """Re-encode the rendered video, optionally muxing audio back in.

    OpenCV's VideoWriter emits MPEG-4 Part 2, which most players and every
    browser refuse. This is the step that makes the output shareable.

    The rendered frames are silent by construction — OpenCV has no concept of an
    audio track, and the MP4 that went to the model was built with ``-an``
    because audio is dead weight in a base64 upload. So the audio has to come
    from *audio_from*, the original source, and this is the only place in the
    pipeline where the two are in the same command.

    Sync needs no correction: the render writes exactly one frame per decoded
    frame at the source's own rate, so both streams start at t=0 and run at the
    same speed. ``-shortest`` trims the tail — this clip's audio stream is 15 ms
    longer than its video, which is normal for a phone recording.

    *audio_codec* defaults to ``"copy"``: the source track is already AAC, and
    re-encoding a 48 kbps stream to hand a lossy copy to the muxer would only
    lose quality. Pass a real encoder (``"aac"``) for a source whose codec MP4
    cannot hold.
    """
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src)]

    if audio_from is not None:
        # -t before -i applies to that input, so a trimmed render gets an
        # equally trimmed audio track rather than the whole song.
        if trim_seconds:
            cmd += ["-t", str(trim_seconds)]
        cmd += ["-i", str(audio_from)]

    cmd += ["-map", "0:v:0"]
    if audio_from is not None:
        cmd += ["-map", "1:a:0"]

    cmd += ["-c:v", "libx264", "-preset", "fast", "-crf", str(crf), "-pix_fmt", "yuv420p"]
    if audio_from is not None:
        cmd += ["-c:a", audio_codec, "-shortest"]
    cmd += [str(dst)]

    _run(cmd)
