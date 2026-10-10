"""Run ViTPose on a local dance video and render the poses back onto it.

    python main.py
    python main.py --li "Your Name" --x @yourhandle   # with your credit in the corner

Everything else is configured in config.py; the flags set only the credit.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from dataclasses import asdict
from datetime import datetime

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

import config as cfg
from src import panel as panelmod
from src import pictures as picturemod
from src import pose, render, similarity, timing, video
from src.env import load_api_key

console = Console()

# The terminal answers "did it run"; report.txt answers "why does this result
# look like that". The diagnostic tables are written to this second console
# instead of the terminal, in the order they happen, and land in the run
# directory at the end. Fixed width so the file does not depend on the terminal
# it was run in, and no color so it reads as plain text.
_report = io.StringIO()
report_console = Console(file=_report, width=100, no_color=True, highlight=False,
                         emoji=False)


def rel(path):
    """Project-relative path for console output; absolute when outside the project."""
    try:
        return path.relative_to(cfg.PROJECT_DIR)
    except ValueError:
        return path


def rule(step: int, title: str) -> None:
    console.rule(f"[bold cyan]{step}[/] {title}", align="left")
    report_console.rule(f"{step} {title}", align="left")


def _kv_table(pairs: dict) -> Table:
    table = Table.grid(padding=(0, 2))
    table.add_column(style="dim", justify="right")
    table.add_column()
    for key, value in pairs.items():
        table.add_row(key, str(value))
    return table


def kv(pairs: dict, title: str | None = None) -> None:
    table = _kv_table(pairs)
    console.print(Panel(table, title=title, title_align="left", expand=False) if title else table)


def say(*args, **kwargs) -> None:
    """Terminal and report — a line of the run's narrative."""
    console.print(*args, **kwargs)
    report_console.print(*args, **kwargs)


def detail(*args, **kwargs) -> None:
    """report.txt only — a diagnostic, never the terminal."""
    report_console.print(*args, **kwargs)


def detail_kv(pairs: dict, title: str | None = None) -> None:
    table = _kv_table(pairs)
    report_console.print(
        Panel(table, title=title, title_align="left", expand=False) if title else table)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Every setting lives in config.py; these override the credit per run.")
    credit = parser.add_argument_group(
        "credit", "the lines in the corner; any of these replaces config.py's CREDIT")
    credit.add_argument("--li", metavar="TEXT", help='LinkedIn, e.g. "Jeremy Park, PhD"')
    credit.add_argument("--x", metavar="HANDLE", help="X handle, e.g. @jeremyparkphd")
    credit.add_argument("--ig", metavar="HANDLE", help="Instagram handle")
    credit.add_argument("--no-credit", action="store_true", help="no credit at all")
    return parser.parse_args(argv)


def apply_args(args: argparse.Namespace) -> None:
    """Write the flags over the config, before anything reads it."""
    def at(handle):
        return handle if handle.startswith("@") else f"@{handle}"

    handles = [(label, value) for label, value in
               (("LI", args.li), ("X", args.x and at(args.x)),
                ("IG", args.ig and at(args.ig))) if value]
    if args.no_credit:
        cfg.CREDIT = []
    elif handles:
        cfg.CREDIT = handles


def main(argv: list[str] | None = None) -> int:
    apply_args(parse_args(argv))
    t_start = time.perf_counter()
    console.print()
    console.rule("[bold]ViTPose — dance[/]", align="center")
    report_console.rule("ViTPose — dance", align="center")

    # ── 1. Key ───────────────────────────────────────────────────────────────
    rule(1, "API key")
    api_key, source = load_api_key(cfg.PROJECT_DIR)
    console.print(f"  loaded from [green]"
                  f"{source if source == 'environment' else 'the nearest .env'}[/]")
    detail("  key loaded from "
           + (source if source == "environment"
              else os.path.relpath(source, cfg.PROJECT_DIR)))

    # ── 2. Source ────────────────────────────────────────────────────────────
    rule(2, "Source video")
    if not cfg.INPUT_VIDEO.is_file():
        console.print(f"[red]Not found:[/] {cfg.INPUT_VIDEO}")
        return 1

    src_info = video.probe_source(cfg.INPUT_VIDEO)
    audio = video.probe_audio(cfg.INPUT_VIDEO) if cfg.KEEP_AUDIO else None
    console.print(
        f"  [bold]{cfg.INPUT_VIDEO.name}[/] — {src_info['codec']} "
        f"{src_info['width']}x{src_info['height']}, "
        f"{src_info['n_frames']} frames, {src_info['duration']:.1f}s"
        + (f", rotation {src_info['rotation']}°" if src_info["rotation"] else "")
        + (f", {audio['codec']} {audio['channels']}ch" if audio
           else "  [yellow]no audio track[/]" if cfg.KEEP_AUDIO else "")
    )
    detail_kv({
        "path": rel(cfg.INPUT_VIDEO),
        "size": f"{cfg.INPUT_VIDEO.stat().st_size / 1e6:.1f} MB",
        "codec": f"{src_info['codec']} ({src_info['pix_fmt']})",
        "stored": f"{src_info['width']}x{src_info['height']}",
        "rotation": f"{src_info['rotation']}°"
                    + ("  [dim](applied on convert; OpenCV would ignore it)[/]"
                       if src_info["rotation"] else ""),
        "frames": f"{src_info['n_frames']} ({src_info['duration']:.1f}s)",
        "audio": (f"{audio['codec']} {audio['channels']}ch {audio['sample_rate'] / 1000:.1f} kHz"
                  + (f" {audio['bit_rate'] // 1000} kbps" if audio["bit_rate"] else "")
                  + f", {audio['duration']:.1f}s  [dim](muxed into the export)[/]")
                 if audio else ("[yellow]none[/] — the export will be silent"
                                if cfg.KEEP_AUDIO else "[dim]not kept (KEEP_AUDIO)[/]"),
    })

    # ── 3. Convert (cached) ──────────────────────────────────────────────────
    rule(3, "Convert to MP4")
    def prepare(height, label):
        """Convert to *height* (None = source), reusing a cached copy if present."""
        path = video.cache_path(
            cfg.INPUT_VIDEO, cfg.CACHE_DIR,
            target_height=height, trim_seconds=cfg.TRIM_SECONDS, crf=cfg.CONVERT_CRF,
        )
        seconds, cached = 0.0, True
        if path.is_file() and not cfg.FORCE_RECONVERT:
            detail(f"  [green]cache hit[/] ({label}) — "
                   f"[dim]{rel(path)}[/]")
        else:
            cached = False
            reason = "FORCE_RECONVERT" if path.is_file() else "no cached copy"
            detail(f"  [yellow]converting[/] ({label}, {reason})")
            t0 = time.perf_counter()
            video.convert(
                cfg.INPUT_VIDEO, path,
                target_height=height, trim_seconds=cfg.TRIM_SECONDS,
                crf=cfg.CONVERT_CRF, total_frames=src_info["n_frames"], console=console,
            )
            seconds = time.perf_counter() - t0
            detail(f"  done in {seconds:.1f}s -> "
                   f"[dim]{rel(path)}[/]")
        return path, video.inspect(path), seconds, cached

    # Two renditions, because they answer to different limits. The model caps at
    # a 2048px long edge internally, so uploading more than that is pure wait for
    # identical poses; the export answers only to what you want to watch, and
    # sets the panel's height with it. Pose coordinates are normalized, so one
    # inference serves any render size — and when the two heights match, the
    # cache key matches too and this converts once.
    mp4, info, convert_seconds, convert_cached = prepare(cfg.INFERENCE_HEIGHT, "inference")

    if cfg.EXPORT_HEIGHT == cfg.INFERENCE_HEIGHT:
        export_mp4, export_info = mp4, info
    else:
        export_mp4, export_info, extra_s, extra_cached = prepare(cfg.EXPORT_HEIGHT, "export")
        convert_seconds += extra_s
        convert_cached = convert_cached and extra_cached
        if export_info.n_frames != info.n_frames:
            say(
                f"  [yellow]warning[/] frame counts differ "
                f"({info.n_frames} inference vs {export_info.n_frames} export); "
                "poses are matched by frame index, so the overlay may drift."
            )

    say(f"  inference [bold]{info}[/]  ({mp4.stat().st_size / 1e6:.1f} MB)"
        + ("  [dim]cached[/]" if convert_cached else ""))
    if export_mp4 is not mp4:
        say(f"  export    [bold]{export_info}[/]  "
            f"({export_mp4.stat().st_size / 1e6:.1f} MB)")

    blank = video.blank_frames(mp4, max_luma=cfg.BLANK_FRAME_MAX_LUMA) \
        if cfg.SKIP_BLANK_FRAMES else []
    if blank:
        say(
            f"  [yellow]{len(blank)} blank frame(s)[/]: {blank if len(blank) <= 8 else blank[:8]}"
            f"{' …' if len(blank) > 8 else ''} — no pixel above "
            f"{cfg.BLANK_FRAME_MAX_LUMA}/255, so nothing can be posed on them"
        )

    # ── 4. Request ───────────────────────────────────────────────────────────
    rule(4, "Pose estimation")
    video_b64, extra_body = pose.build_request(
        mp4,
        every_frame=cfg.EVERY_FRAME, fps=info.fps, n_frames=info.n_frames,
        video_fps=cfg.VIDEO_FPS, video_max_frames=cfg.VIDEO_MAX_FRAMES,
        precision=cfg.PRECISION,
    )
    billed = extra_body.get("video_max_frames") or min(info.n_frames, 900)
    console.print(
        f"  [bold]{cfg.MODEL}[/] — "
        + ("every frame" if cfg.EVERY_FRAME else f"video_fps={cfg.VIDEO_FPS}")
        + f", ~{billed} frames billed, {len(video_b64) / 1e6:.1f} MB upload"
    )
    detail_kv({
        "model": cfg.MODEL,
        "mode": ("every frame — detector stride 1, nothing skipped"
                 if cfg.EVERY_FRAME else f"video_fps={cfg.VIDEO_FPS} (detector cadence)"),
        "video_fps": extra_body["video_fps"],
        "video_max_frames": extra_body.get("video_max_frames", "— (model default: 900)"),
        "billed units": f"~{extra_body.get('video_max_frames') or min(info.n_frames, 900)} frames",
        "upload": f"{len(video_b64) / 1e6:.1f} MB base64",
    }, title="request")

    from openai import OpenAI

    poses_cache = pose.cache_path(mp4, cfg.CACHE_DIR, model=cfg.MODEL,
                                  extra_body=extra_body)
    cached = pose.load_cache(poses_cache) if cfg.REUSE_POSES else None

    if cached is not None:
        # Reuse is deliberate and loud: rendering changes should not cost a
        # model call, but a silent cache hit would make a stale result look
        # fresh. The key covers the clip and every request field, so a hit
        # cannot mean "different settings".
        payload, usage, cached_timing, created = cached
        request_timing = timing.RequestTiming(**cached_timing) if cached_timing \
            else timing.RequestTiming()
        elapsed = request_timing.round_trip
        say(f"  [green]cache hit[/] — poses from [dim]{created}[/], no gateway call")
        detail(f"  [dim]{poses_cache.name}; set REUSE_POSES = False to re-run detection[/]")
    else:
        if cfg.REUSE_POSES:
            say("  [yellow]no cached poses for these settings[/] — calling the gateway")

        marks: dict = {}
        http_client = timing.timed_http_client(cfg.REQUEST_TIMEOUT, marks)
        client = OpenAI(
            base_url=cfg.GATEWAY_BASE_URL, api_key=api_key,
            timeout=cfg.REQUEST_TIMEOUT, max_retries=1,
            **({"http_client": http_client} if http_client else {}),
        )

        t0 = time.perf_counter()
        with console.status("[cyan]waiting on the gateway[/] — pose runs on every decoded frame…",
                            spinner="dots"):
            payload, usage = pose.request_poses(
                client, model=cfg.MODEL, video_b64=video_b64, extra_body=extra_body
            )
        elapsed = time.perf_counter() - t0
        request_timing = timing.summarize(marks, elapsed)

        pose.save_cache(poses_cache, payload, usage, asdict(request_timing),
                        stamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    frames, by_index = pose.unwrap(payload)

    # One box per real subject, before anything else reads the poses. The raw
    # response carries ten track ids for three dancers, and the overlay is only
    # as good as this pass.
    report = pose.clean(
        frames,
        blank_indices=set(blank),
        min_box_area=cfg.MIN_BOX_AREA, drop_frozen=cfg.DROP_FROZEN_BOXES,
        min_track_fraction=cfg.MIN_TRACK_FRACTION, min_track_area=cfg.MIN_TRACK_AREA,
        max_persons=cfg.MAX_PERSONS, iou=cfg.PERSON_IOU_THRESHOLD,
        merge_distance=cfg.TRACK_MERGE_DISTANCE,
        merge_min_frames=cfg.TRACK_MERGE_MIN_FRAMES,
    )
    by_index = {f["index"]: f["persons"] for f in frames}
    result = pose.PoseResult(payload, frames, by_index, elapsed, usage)

    console.print(f"  [bold]{len(report.kept_track_ids)} subjects[/] "
                  f"[dim]from {len(report.tracks)} track ids"
                  + (f", {report.merged} detections merged" if report.merges else "")
                  + "[/]")

    if cfg.REPORT_TRACKS:
        table = Table(box=None, pad_edge=False)
        for col, justify in (("track", "right"), ("frames", "right"),
                             ("lifetime", "right"), ("median area", "right"),
                             ("", "left")):
            table.add_column(col, justify=justify)
        for row in report.tracks:
            verdict = "[green]person[/]" if row["kept"] else f"[dim]ghost — {row['reason']}[/]"
            table.add_row(str(row["track_id"]), str(row["frames"]),
                          f"{row['lifetime']:.2f}", f"{row['median_area']:.4f}", verdict)
        detail(Panel(
            table,
            title=f"[bold]{len(report.kept_track_ids)} subjects[/] "
                  f"[dim]from {len(report.tracks)} track ids[/]",
            title_align="left", expand=False))
        for keep, absorb, distance in report.merges:
            detail(f"  [green]merged[/] track {absorb} into {keep} — "
                   f"the same body under two ids "
                   f"[dim](median keypoint distance {distance:.4f})[/]")

    console.print(f"  cleanup: [bold]{report.before} -> {report.after}[/] detections")
    detail_kv({
        "detections": f"[bold]{report.before} -> {report.after}[/]",
        "on blank frames": report.blank,
        "degenerate": report.degenerate,
        "repeated track id": report.same_track,
        "too small": report.tiny,
        "frozen": report.frozen,
        "ghost-track": report.ghost_tracks,
        "merged": report.merged,
        "handoff duplicate": report.duplicates,
        **({"over MAX_PERSONS": report.capped} if report.capped else {}),
    }, title="cleanup")

    coverage = len(by_index) / info.n_frames if info.n_frames else 0
    if cached is not None:
        pass
    elif request_timing.measured_split:
        console.print(
            f"  [green]done in {elapsed:.1f}s[/] — "
            f"upload {request_timing.upload:.1f}s, "
            f"[bold]server {request_timing.server:.1f}s[/], "
            f"download {request_timing.download:.1f}s"
        )
    else:
        console.print(f"  [green]done in {elapsed:.1f}s[/]")

    per_frame = [len(p) for p in by_index.values()]
    console.print(
        f"  [bold]{len(frames)}[/] of {info.n_frames} frames returned "
        f"[dim]({coverage:.0%})[/], "
        + (f"max {max(per_frame)} people, mean {sum(per_frame) / len(per_frame):.2f}"
           if per_frame else "no people")
        + f"  [dim]tracks {result.track_ids or 'none'}[/]"
    )
    detail_kv({
        "frames returned": f"{len(frames)} of {info.n_frames} source ({coverage:.0%})",
        "frames with a person": result.n_posed,
        "people per frame": f"max {max(per_frame or [0])}, "
                            f"mean {sum(per_frame) / len(per_frame):.2f}" if per_frame else "—",
        "track ids": result.track_ids or "none",
        "usage": usage or "—",
    }, title="response")

    if cfg.EVERY_FRAME and len(by_index) < info.n_frames:
        say(
            f"  [yellow]note[/] {info.n_frames - len(by_index)} frames were not returned "
            "despite EVERY_FRAME; the decoder's frame count can differ slightly from OpenCV's."
        )

    # ── 5. Output ────────────────────────────────────────────────────────────
    rule(5, "Render")
    stamp = datetime.now().strftime(cfg.RUN_STAMP_FORMAT)
    run_dir = cfg.OUTPUT_DIR / stamp
    run_dir.mkdir(parents=True, exist_ok=True)

    poses_path = run_dir / "poses.json"
    poses_path.write_text(json.dumps(payload, indent=2))

    sim = None
    pics = None
    if cfg.SIDE_PANEL or cfg.SAVE_SIMILARITY_PLOT:
        sim = similarity.analyze(frames, width=info.width, height=info.height,
                                 fps=info.fps, n_frames=info.n_frames, cfg=cfg)

        # Pictures are resolved before anything reports, because when they are
        # the metric they rewrite the very fields the reporting reads.
        pics = picturemod.analyze(sim, cfg=cfg) if (sim.n_scored and cfg.PICTURES) else None
        if pics is not None and pics.n and cfg.PICTURE_WEIGHTED:
            # Pictures inform and weight the metric; they do not replace it.
            # The per-frame series stays continuous so the trace can be read
            # against a timestamp.
            picturemod.contribute(sim, pics, cfg=cfg)

        if sim.n_scored:
            sim_stats = sim.summary()
            sp = sim_stats["score_percent"]
            console.print(
                f"  sync score    [bold]{sp['picture_weighted_mean']:.0f}%[/] "
                f"picture-weighted  [dim]({sp['mean']:.0f}% flat, "
                f"{sp['min']:.0f}-{sp['max']:.0f}% range, "
                f"{sim.n_scored} of "
                f"{sim.window_frames[1] - sim.window_frames[0] + 1} frames scored)[/]"
            )
            if sim.components:
                console.print(
                    "  components    " + "   ".join(
                        f"{c.name} [bold]{c.summary()['score_percent']['median']:.0f}%[/]"
                        f" [dim]x{c.weight:.2f}[/]" for c in sim.components)
                )
            if sim.ranking:
                ranks = {row["rank"] for row in sim.ranking}
                console.print(
                    "  most in sync  " + ", ".join(
                        f"d{row['dancer']} {row['mean']:.1f}%"
                        for row in sorted(sim.ranking, key=lambda r: r["rank"]))
                    + ("  [dim](all tied)[/]" if len(ranks) == 1 else "")
                )
            detail_kv({
                "window": (f"{cfg.SCORE_WINDOW[0]}s - {cfg.SCORE_WINDOW[1]}s "
                           f"[dim](frames {sim.window_frames[0]}-{sim.window_frames[1]}, "
                           f"{(sim.window_frames[1] - sim.window_frames[0] + 1) / info.fps:.1f}s "
                           f"of {info.duration:.1f}s; the null is measured inside it too)[/]")
                          if cfg.SCORE_WINDOW else "[dim]whole clip[/]",
                "scored": f"{sim.n_scored} of "
                          f"{sim.window_frames[1] - sim.window_frames[0] + 1} frames in window "
                          f"— all {sim.n_subjects} dancers fully in shot",
                "excluded": ", ".join(f"{v} {k.replace('_', ' ')}"
                                      for k, v in sim.excluded.items() if v) or "none",
                "posture from": f"[bold]{sim.metric}[/] — "
                                + ("limb directions in the torso frame"
                                   if sim.metric == "angles" else
                                   "joint positions, torso-length normalized"),
                "weight share": "  ".join(
                    f"[dim]{k}[/] {v:.0f}%" for k, v in
                    sorted(sim.feature_share.items(), key=lambda kv: -kv[1])),
                "source": f"[bold]{sim.source}[/]",
                "summary": f"[bold]{sim_stats['score_percent']['picture_weighted_mean']:.0f}% "
                           f"picture-weighted average[/], "
                           f"{sim_stats['score_percent']['mean']:.0f}% flat, "
                           f"{sim_stats['score_percent']['median']:.0f}% median, "
                           f"{sim_stats['score_percent']['min']:.0f}-"
                           f"{sim_stats['score_percent']['max']:.0f}% range",
                "per pair": "   ".join(f"[dim]{a}v{b}[/] {v['median']:.0f}%"
                                       for (a, b), v in zip(sim.pair_labels,
                                                            sim_stats["pairs"].values())),
            }, title="sync score")

            ctab = Table(box=None, pad_edge=False)
            for col in ("component", "weight", "0% anchor", "100% anchor",
                        "matched", "median", "leaderboard"):
                ctab.add_column(col, justify="right" if col != "component" else "left")
            for comp in sim.components:
                cs = comp.summary()["score_percent"]
                order = sorted(comp.ranking, key=lambda r: r["rank"])
                board = "  ".join(
                    f"[dim]#{r['rank']}[/] d{r['dancer']} {r['mean']:.0f}%" for r in order)
                ctab.add_row(
                    f"[bold]{comp.name}[/]", f"{comp.weight:.2f}",
                    f"{comp.null_distance:.1f} {comp.unit}",
                    f"<{comp.tolerance:.1f}",
                    f"{comp.summary()['median_matched_distance']:.1f}",
                    f"{cs['median']:.0f}%", board,
                )
            formula = " + ".join(f"{c.weight:.2f}x{c.name}" for c in sim.components)
            blended = any(getattr(c, "picture_blended", False) for c in sim.components)
            title = (f"[bold]summary = {formula}[/] [dim](timing crossfaded to the "
                     f"picture arrivals near each landing)[/]" if blended else
                     f"[bold]summary = {formula}[/] [dim](component percentages, each "
                     f"null-calibrated)[/]")
            detail(Panel(ctab, title=title, title_align="left", expand=False))

            table = Table(box=None, pad_edge=False)
            for col in ("rank", "dancer", "mean", "95% CI", "ranks 1st", ""):
                table.add_column(col, justify="right" if col != "" else "left")
            for row in sim.ranking:
                ci = row.get("ci95")
                note = ("[yellow]tied with " +
                        ", ".join(str(t) for t in row["tied_with"]) + "[/]"
                        ) if row["tied_with"] else ""
                table.add_row(
                    f"[bold]{row['rank']}[/]", str(row["dancer"]),
                    f"{row['mean']:.1f}%",
                    f"{ci[0]:.1f}-{ci[1]:.1f}" if ci else "—",
                    f"{row['p_ranks_first'] * 100:.0f}%" if "p_ranks_first" in row else "—",
                    note,
                )
            detail(Panel(
                table, title="[bold]most in sync with the others[/] "
                             "[dim](block bootstrap, ties shared)[/]",
                title_align="left", expand=False))
        else:
            say("  [yellow]no similarity score[/] — no frame had every "
                          "subject fully in shot")
            sim = None

    if pics is not None and pics.n:
        ps = pics.summary()
        console.print(
            f"  pictures      [bold]{pics.n}[/] matched across all "
            f"{sim.n_subjects} dancers  [dim]"
            f"{ps['score_percent']['median']:.0f}% median score, "
            f"{ps['timing_spread_ms']['median']:.0f} ms median spread[/]"
        )
        detail_kv({
            "pictures": f"[bold]{pics.n}[/] matched across all {sim.n_subjects} dancers "
                        f"[dim](detected "
                        + ", ".join(f"d{k} {v}" for k, v in ps["detected_per_dancer"].items())
                        + f"; {ps['unmatched_detections']} unmatched)[/]",
            "shape at the landing": f"[bold]{ps['shape_percent']['median']:.0f}% median[/], "
                                    f"{ps['shape_percent']['min']:.0f}-"
                                    f"{ps['shape_percent']['max']:.0f}% range "
                                    f"[dim](each dancer read at their own arrival)[/]",
            "picture score": (f"[bold]{ps['score_percent']['median']:.0f}% median[/] "
                              f"[dim]= {ps['construction']['formula']}; "
                              f"{ps['score_percent']['min']:.0f}-"
                              f"{ps['score_percent']['max']:.0f}% range[/]"),
            "timing of the landing": f"[bold]{ps['timing_spread_ms']['median']:.0f} ms median "
                                     f"spread[/], p90 {ps['timing_spread_ms']['p90']:.0f} ms, "
                                     f"worst {ps['timing_spread_ms']['max']:.0f} ms "
                                     f"[dim]({ps['timing_spread_ms']['share_under_50ms']:.0%} "
                                     f"under 50 ms)[/]",
            "who drifts": "   ".join(
                f"[dim]d{k}[/] {v:+.0f} ms" for k, v in ps["per_dancer_offset_ms"].items()),
        }, title="pictures")

        worst = sorted(pics.pictures, key=lambda p: p.score_percent)[:3]
        loose = sorted(pics.pictures, key=lambda p: -p.spread_ms)[:3]
        table = Table(box=None, pad_edge=False)
        for col in ("", "at", "score", "shape", "spread", "offsets (ms)"):
            table.add_column(col, justify="right" if col else "left")
        for tag, group in (("[yellow]worst overall[/]", worst),
                           ("[yellow]loosest timing[/]", loose)):
            for i, pic in enumerate(group):
                table.add_row(tag if i == 0 else "", f"{pic.time_seconds:.1f}s",
                              f"[bold]{pic.score_percent:.0f}%[/]",
                              f"{pic.shape_percent:.0f}%", f"{pic.spread_ms:.0f} ms",
                              " ".join(f"{v:+.0f}" for v in pic.offsets_ms.values()))
        detail(Panel(table, title="[bold]pictures to look at[/]",
                     title_align="left", expand=False))

    elif pics is not None:
        say("  [yellow]no pictures matched across all dancers[/]")

    raw = run_dir / "_raw.mp4"
    t0 = time.perf_counter()
    stats = render.render(export_info, by_index, len(frames), raw, cfg=cfg, console=console,
                          similarity=sim,
                          picture_times=[p.time_seconds for p in pics.pictures]
                          if pics is not None else ())
    render_seconds = time.perf_counter() - t0
    say(f"  drew poses on [bold]{stats['frames_with_pose']}/{stats['frames_written']}[/] "
        f"frames" + (f" ({stats['frames_held']} held)" if stats["frames_held"] else ""))
    detail_kv({
        "output": f"{stats['output_width']}x{export_info.height}"
                  + ("  (clip + equal-width panel)" if stats["side_panel"] else "  (clip only)"),
        "panel font": stats["panel_font"] or "no panel",
        "frames with a pose": f"{stats['frames_with_pose']} of {stats['frames_written']}"
                              + (f", {stats['frames_held']} held across a decimation gap"
                                 if stats["frames_held"] else ""),
        "people drawn": f"max {stats['max_people_in_a_frame']}, "
                        f"mean {stats['mean_people_per_posed_frame']} per posed frame",
    }, title="render")

    out_video = run_dir / f"{cfg.INPUT_VIDEO.stem}_pose.mp4"
    t0 = time.perf_counter()
    video.encode_h264(
        raw, out_video, crf=cfg.OUTPUT_CRF,
        audio_from=cfg.INPUT_VIDEO if audio else None,
        audio_codec=cfg.AUDIO_CODEC, trim_seconds=cfg.TRIM_SECONDS,
    )
    encode_seconds = time.perf_counter() - t0
    raw.unlink(missing_ok=True)   # MPEG-4 Part 2 intermediate; no player wants it
    say(f"  video  -> [dim]{rel(out_video)}[/] "
                  f"({out_video.stat().st_size / 1e6:.1f} MB"
                  + (", audio copied from the source" if audio and cfg.AUDIO_CODEC == "copy"
                     else f", audio re-encoded to {cfg.AUDIO_CODEC}" if audio else ", silent")
                  + ")")

    sim_path = sim_plot = comp_plot = None
    if sim is not None:
        sim_path = run_dir / "similarity.json"
        sim_path.write_text(json.dumps({
            "definition": {
                "kind": sim.metric,
                "unit": sim.unit,
                "features": sim.feature_names,
                "weights": {k: v for k, v in (cfg.ANGLE_WEIGHTS if sim.metric == "angles"
                                              else cfg.POSITION_WEIGHTS).items()},
                "weight_mode": cfg.ANGLE_WEIGHT_MODE if sim.metric == "angles" else "raw",
                "weight_share_percent": sim.feature_share,
                "normalization": ("segment directions in the dancer's own torso frame, "
                                  "isotropic pixel units" if sim.metric == "angles" else
                                  f"mid-hip origin, {cfg.SCALE_MODE} torso-length scale, "
                                  "isotropic pixel units"),
                "score_window_seconds": list(cfg.SCORE_WINDOW) if cfg.SCORE_WINDOW else None,
                "null_lag_seconds": cfg.NULL_LAG_SECONDS,
                "null_distance": round(sim.null_distance, 5),
                "tolerance": round(sim.tolerance, 5),
                "noise_floor": round(sim.noise_floor, 5),
                "component_score": ("100 * clamp(1 - (difference - tolerance) "
                                    "/ (null_distance - tolerance), 0, 1)"),
                "summary_weights": dict(cfg.SUMMARY_WEIGHTS),
                "timing": {
                    "signal": "per-segment angular speed, magnitude only (deg/s)",
                    "why_separate": ("frame-by-frame pose comparison conflates shape "
                                     "error with timing error: during a fast move a "
                                     "3-frame lag reads as a large pose difference, "
                                     "during a held pause the same lag reads as none"),
                    "smooth_frames": cfg.TIMING_SMOOTH_FRAMES,
                    "tolerance": cfg.TIMING_TOLERANCE,
                },
                "smooth_frames": cfg.SIMILARITY_SMOOTH_FRAMES,
                "interpolate_max_gap_frames": cfg.SIMILARITY_INTERPOLATE_MAX_GAP,
                "panel_refresh_frames": cfg.PANEL_REFRESH_FRAMES,
                "per_dancer": "mean of the pairs containing that dancer — "
                              "agreement with the others, not correctness",
                "per_dancer_note": ("mean pairwise squared difference equals 2n/(n-1) times "
                                    "the mean squared difference from the group consensus, "
                                    "so averaging pairs is a group measure, not a shortcut"),
                "ranking": {
                    "method": "block bootstrap over the per-dancer series",
                    "block_seconds": cfg.RANK_BOOTSTRAP_BLOCK_SECONDS,
                    "samples": cfg.RANK_BOOTSTRAP_SAMPLES,
                    "seed": cfg.RANK_BOOTSTRAP_SEED,
                    "tie_band": cfg.RANK_TIE_BAND,
                },
            },
            **sim.summary(),
            "pair_labels": [list(p) for p in sim.pair_labels],
            "slots": sim.slots,
            "series": {
                "frame": sim.indices.tolist(),
                "time": [round(float(t), 4) for t in sim.times()],
                "score": [round(float(v), 3) for v in sim.score],
                "pairwise": [[round(float(v), 3) for v in row] for row in sim.pairwise],
                "per_dancer": [[round(float(v), 3) for v in row]
                               for row in sim.dancer_series[sim.indices]],
                "posture": [round(float(v), 3)
                            for v in sim.components[0].score] if sim.components else [],
                "timing": [round(float(v), 3)
                           for v in sim.components[1].score] if len(sim.components) > 1 else [],
                "per_dancer_posture": [[round(float(v), 3) for v in row]
                                       for row in sim.components[0].dancer_series[sim.indices]]
                                      if sim.components else [],
                "per_dancer_timing": [[round(float(v), 3) for v in row]
                                      for row in sim.components[1].dancer_series[sim.indices]]
                                     if len(sim.components) > 1 else [],
                "distance": [[round(float(v), 5) for v in row] for row in sim.distance],
            },
        }, indent=2))
    
        if cfg.SAVE_SIMILARITY_PLOT:
            sim_plot = run_dir / "similarity.png"
            st = sim.summary()["score_percent"]
            panelmod.plot_similarity(
                sim, sim_plot, cfg=cfg,
                title=f"Sync score — {sim.n_subjects} dancers, "
                      f"{sim.n_scored} scored frames, median {st['median']:.0f}%",
            )

        if pics is not None and pics.n:
            pic_path = run_dir / "pictures.json"
            pic_path.write_text(json.dumps({
                "definition": {
                    "picture": "a local minimum of that dancer's own angular-speed "
                               "envelope — the moment they stop moving",
                    "shape": "posture difference with each dancer read AT THEIR OWN "
                             "arrival frame, on the same scale as the continuous "
                             "posture score",
                    "timing": "spread of the arrival times, in milliseconds",
                    "min_gap_frames": cfg.PICTURE_MIN_GAP_FRAMES,
                    "prominence": cfg.PICTURE_PROMINENCE,
                    "cluster_frames": cfg.PICTURE_CLUSTER_FRAMES,
                },
                **pics.summary(),
                "pictures": [p.as_dict() for p in pics.pictures],
            }, indent=2))

        if cfg.SAVE_COMPONENT_PLOT and sim.components:
            comp_plot = run_dir / "components.png"
            panelmod.plot_components(
                sim, comp_plot, cfg=cfg,
                title=f"Posture vs timing — {sim.n_subjects} dancers, "
                      f"{sim.n_scored} scored frames",
            )

        if cfg.SAVE_RANKING_PLOT and sim.ranking:
            rank_plot = run_dir / "ranking.png"
            ranks = {row["rank"] for row in sim.ranking}
            panelmod.plot_ranking(
                sim, rank_plot, cfg=cfg,
                title=f"Per-dancer ranking — {sim.n_subjects} dancers, "
                      f"{sim.n_scored} scored frames"
                      + (", all tied" if len(ranks) == 1 else ""),
            )

    plot_path = None
    if cfg.SAVE_JOINT_PLOT:
        plot_path = run_dir / f"{cfg.PLOT_JOINT}.png"
        if not render.plot_joint(frames, info.fps, cfg.PLOT_JOINT, plot_path):
            say(f"  [yellow]plot skipped[/] — {cfg.PLOT_JOINT} never visible")
            plot_path = None

    # ── 6. Metrics ───────────────────────────────────────────────────────────
    rule(6, "Performance")
    total = time.perf_counter() - t_start
    metrics = timing.Metrics(
        frames=len(frames),
        video_seconds=info.duration,
        convert_seconds=convert_seconds,
        convert_cached=convert_cached,
        encode_seconds=encode_seconds,
        request=request_timing,
        render_seconds=render_seconds,
        total_seconds=total,
        upload_mb=len(video_b64) / 1e6,
        cost_usd=(usage or {}).get("cost"),
        poses_cached=cached is not None,
    )

    (run_dir / "metrics.json").write_text(json.dumps(metrics.as_dict(), indent=2))
    (run_dir / "metrics.txt").write_text(metrics.as_text())

    m = metrics.as_dict()
    kv({
        "ViTPose throughput": f"[bold green]{m['pose']['fps']:.1f} fps[/]  "
                              f"({m['pose']['ms_per_frame']:.0f} ms/frame, "
                              f"{m['pose']['realtime_factor']:.2f}x realtime)",
        "server time": f"{m['pose']['seconds']:.2f}s for {m['frames']} frames",
        "gateway round trip": f"{m['gateway_round_trip']['seconds']:.2f}s "
                              f"({m['gateway_round_trip']['fps']:.1f} fps end-to-end)",
        "  upload": f"{m['gateway_round_trip']['upload_seconds']:.2f}s "
                    f"({m['gateway_round_trip']['upload_mb']:.1f} MB)"
                    if m["gateway_round_trip"]["upload_seconds"] is not None else "not separable",
        "render": f"{m['local']['render_seconds']:.2f}s ({m['local']['render_fps']:.1f} fps)",
        "cost": f"${m['cost']['usd']:.6f}  "
                f"(${m['cost']['usd_per_1000_frames']:.4f} / 1000 frames)"
                if "cost" in m else "—",
    }, title="metrics")
    detail(metrics.as_text(), markup=False, highlight=False)

    (run_dir / "run.json").write_text(json.dumps({
        "stamp": stamp,
        "total_seconds": round(total, 2),
        "input": {"path": str(cfg.INPUT_VIDEO), **src_info, "audio": audio},
        "converted": {"path": str(mp4), "width": info.width, "height": info.height,
                      "export_path": str(export_mp4), "export_width": export_info.width,
                      "export_height": export_info.height,
                      "fps": info.fps, "n_frames": info.n_frames},
        "request": {"model": cfg.MODEL, "extra_body": extra_body,
                    "upload_bytes_b64": len(video_b64)},
        "response": {"elapsed_seconds": round(elapsed, 2), "frames_returned": len(frames),
                     "frames_with_person": result.n_posed, "track_ids": result.track_ids,
                     "usage": usage},
        "cleanup": {"before": report.before, "after": report.after,
                    "blank": report.blank, "blank_frames": blank,
                    "degenerate": report.degenerate, "same_track": report.same_track,
                    "tiny": report.tiny, "frozen": report.frozen,
                    "ghost_tracks": report.ghost_tracks, "merged": report.merged,
                    "merges": [{"kept": k, "absorbed": a, "kpt_distance": d}
                               for k, a, d in report.merges],
                    "duplicates": report.duplicates,
                    "capped": report.capped,
                    "thresholds": {"blank_frame_max_luma": cfg.BLANK_FRAME_MAX_LUMA
                                   if cfg.SKIP_BLANK_FRAMES else None,
                                   "min_box_area": cfg.MIN_BOX_AREA,
                                   "drop_frozen_boxes": cfg.DROP_FROZEN_BOXES,
                                   "min_track_fraction": cfg.MIN_TRACK_FRACTION,
                                   "min_track_area": cfg.MIN_TRACK_AREA,
                                   "iou": cfg.PERSON_IOU_THRESHOLD,
                                   "max_persons": cfg.MAX_PERSONS},
                    "tracks": report.tracks},
        "render": {**stats, "draw_face": cfg.DRAW_FACE, "color_by": cfg.COLOR_BY},
        "metrics": metrics.as_dict(),
        "audio": {"kept": bool(audio), "codec": cfg.AUDIO_CODEC if audio else None},
        "similarity": sim.summary() if sim else None,
        "outputs": {"video": out_video.name, "poses": poses_path.name,
                    "similarity": sim_path.name if sim_path else None,
                    "similarity_plot": sim_plot.name if sim_plot else None,
                    "component_plot": comp_plot.name if comp_plot else None,
                    "plot": plot_path.name if plot_path else None,
                    "metrics": "metrics.json"},
    }, indent=2))

    if cfg.SAVE_REPORT:
        report_console.rule("files", align="left")
        detail_kv({
            **{f.name: (f"{f.stat().st_size / 1e6:.1f} MB" if f.stat().st_size >= 1e6
                        else f"{f.stat().st_size / 1e3:.0f} kB")
               for f in sorted(run_dir.iterdir()) if f.is_file()},
            "report.txt": "this file",
        })
        (run_dir / "report.txt").write_text(
            f"dance — run {stamp}\n"
            f"{cfg.INPUT_VIDEO.name}, {len(frames)} frames, {info.duration:.1f}s\n\n"
            + _report.getvalue()
        )

    console.print()
    console.rule(f"[bold green]done[/] in {total:.1f}s", align="left")
    console.print(f"  [bold]{rel(run_dir)}/[/]")
    console.print("  [dim]"
                  + ", ".join(sorted(f.name for f in run_dir.iterdir() if f.is_file()))
                  + "[/]")
    console.print()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]interrupted[/]")
        sys.exit(130)
    except Exception as exc:
        console.print(f"\n[red]error:[/] {exc}")
        sys.exit(1)
