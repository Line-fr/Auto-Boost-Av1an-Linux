#!/usr/bin/env python3
"""
robust_autocrop.py

Batch-detect crop values for many videos using ffmpeg cropdetect in a robust way.
Modified for Auto-Boost-Av1an integration.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shlex
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

CROP_RE = re.compile(r"\bcrop=(\d+):(\d+):(\d+):(\d+)\b")

VIDEO_DEFAULT_EXTS = {
    ".mp4",
    ".mkv",
    ".mov",
    ".m4v",
    ".webm",
    ".avi",
    ".mpg",
    ".mpeg",
    ".ts",
    ".m2ts",
    ".wmv",
}


@dataclass
class VideoInfo:
    path: Path
    width: int
    height: int
    duration: float  # seconds (may be 0/unknown for some files)


@dataclass
class CropResult:
    crop: str  # "W:H:X:Y"
    w: int
    h: int
    x: int
    y: int
    confidence: float  # 0..1
    samples: int  # number of crops observed
    chosen_from_limits: List[float]  # which cropdetect limits produced this crop
    notes: str


def run_cmd(cmd: List[str], timeout: int = 120) -> subprocess.CompletedProcess:
    # Use explicit encoding handling for windows
    return subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        timeout=timeout,
        check=False,
    )


def ffprobe_info(path: Path) -> Optional[VideoInfo]:
    # Pull width/height/duration for the first video stream
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height:format=duration",
        "-of",
        "json",
        str(path),
    ]
    p = run_cmd(cmd, timeout=60)
    if p.returncode != 0:
        return None

    try:
        data = json.loads(p.stdout)
        streams = data.get("streams", [])
        if not streams:
            return None
        w = int(streams[0].get("width", 0) or 0)
        h = int(streams[0].get("height", 0) or 0)
        dur = float(data.get("format", {}).get("duration", 0) or 0.0)
        if w <= 0 or h <= 0:
            return None
        return VideoInfo(path=path, width=w, height=h, duration=dur)
    except Exception:
        return None


def sample_timestamps(duration: float, n: int) -> List[float]:
    """
    Pick timestamps spread across the video, avoiding very beginning/end.
    If duration is unknown/zero, fall back to a few early timestamps.
    """
    if duration and duration > 10:
        start = max(0.5, duration * 0.05)
        end = max(start + 1.0, duration * 0.95)
        if end <= start:
            return [start]
        return [start + (end - start) * (i / (n - 1)) for i in range(n)]
    # unknown duration
    return [0.5, 3.0, 8.0, 15.0][: max(1, min(n, 4))]


def run_cropdetect_segment(
    video: Path,
    ss: float,
    seg: float,
    fps: float,
    limit: float,
    round_to: int,
) -> List[Tuple[int, int, int, int]]:
    """
    Run ffmpeg cropdetect on a short segment and return all detected crops.
    """
    vf = f"fps={fps},format=yuv444p,cropdetect=limit={limit}:round={round_to}:reset=0"
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "info",
        "-ss",
        f"{ss:.3f}",
        "-i",
        str(video),
        "-t",
        f"{seg:.3f}",
        "-vf",
        vf,
        "-an",
        "-sn",
        "-dn",
        "-f",
        "null",
        "-",
    ]
    p = run_cmd(cmd, timeout=max(60, int(seg * 10) + 30))
    text = (p.stderr or "") + "\n" + (p.stdout or "")
    crops: List[Tuple[int, int, int, int]] = []
    for m in CROP_RE.finditer(text):
        w, h, x, y = map(int, m.groups())
        crops.append((w, h, x, y))
    return crops


def area(w: int, h: int) -> int:
    return w * h


def choose_best_crop(
    vi: VideoInfo,
    observed: Counter,
    crop_to_limits: Dict[str, set],
) -> Optional[CropResult]:
    """
    Choose best crop among observed crops:
    - prefer stability (high count)
    - avoid overcropping: among similar counts, prefer larger area
    - prefer crops supported by multiple limits
    """
    if not observed:
        return None

    full_area = area(vi.width, vi.height)

    # Score: stability + mild preference for larger area + support across limits
    best = None
    best_score = -1e18

    total = sum(observed.values())

    for crop_str, count in observed.items():
        w, h, x, y = map(int, crop_str.split(":"))
        a = area(w, h)
        if a <= 0 or a > full_area:
            continue

        # Fraction of frames/samples that reported this crop
        freq = count / max(1, total)

        # Area ratio: closer to 1 = less cropping (safer)
        ar = a / full_area

        # Limits support: more independent thresholds agreeing = good sign
        lim_support = len(crop_to_limits.get(crop_str, set()))

        # Heuristic score (tuned for “robust but not reckless”):
        # - frequency dominates
        # - larger area breaks ties (avoid overcropping)
        # - multi-limit agreement boosts confidence
        score = (freq * 1000.0) + (ar * 50.0) + (lim_support * 15.0)

        # Slight penalty if crop is exactly full frame (still valid, but less “found bars”)
        if w == vi.width and h == vi.height and x == 0 and y == 0:
            score -= 5.0

        if score > best_score:
            best_score = score
            best = (crop_str, w, h, x, y, freq, total, lim_support)

    if best is None:
        return None

    crop_str, w, h, x, y, freq, total, lim_support = best
    limits = sorted(crop_to_limits.get(crop_str, set()))
    notes = []
    if lim_support >= 3:
        notes.append("strong multi-threshold agreement")
    elif lim_support == 2:
        notes.append("moderate multi-threshold agreement")
    else:
        notes.append("single-threshold pick (still may be correct)")

    return CropResult(crop_str, w, h, x, y, freq, total, limits, ", ".join(notes))


def detect_crop(
    video: Path,
    limits: List[float] = [24 / 255.0, 32 / 255.0],
    round_to: int = 2,
    segments: int = 5,
    segment_duration: float = 2.0,
    check_fps: float = 2.0,
) -> Optional[str]:
    """
    Main entry for single video. Returns "W:H:X:Y" or None.
    """
    vi = ffprobe_info(video)
    if not vi:
        return None

    # Pick timestamps
    timestamps = sample_timestamps(vi.duration, segments)

    observed = Counter()
    crop_to_limits = defaultdict(set)

    for limit in limits:
        for t in timestamps:
            crops = run_cropdetect_segment(
                video, t, segment_duration, check_fps, limit, round_to
            )
            for w, h, x, y in crops:
                cstr = f"{w}:{h}:{x}:{y}"
                observed[cstr] += 1
                crop_to_limits[cstr].add(limit)

    res = choose_best_crop(vi, observed, crop_to_limits)
    if res:
        return res.crop
    return None


def main():
    parser = argparse.ArgumentParser(description="Robust Batch Autocrop")
    parser.add_argument("directory", nargs="?", default=".", help="Directory to scan")
    parser.add_argument(
        "--limit",
        "-l",
        type=float,
        nargs="+",
        default=[24 / 255.0, 30 / 255.0],
        help="Black thresholds",
    )
    parser.add_argument(
        "--round", "-r", type=int, default=2, help="Round dims to multiple of N"
    )
    parser.add_argument("--recursive", "-R", action="store_true", help="Recursive scan")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print crops but don't save"
    )
    args = parser.parse_args()

    # If directories provided... (omitted full CLI implementation for now as this is called by Auto-Boost-Av1an mostly)
    # But just in case user runs it manually:

    root = Path(args.directory).resolve()
    if not root.exists():
        print(f"Error: {root} not found.")
        sys.exit(1)

    videos = []
    if root.is_file():
        videos.append(root)
    else:
        pattern = "**/*" if args.recursive else "*"
        for f in root.glob(pattern):
            if f.suffix.lower() in VIDEO_DEFAULT_EXTS:
                videos.append(f)

    if not videos:
        print("No videos found.")
        return

    print(f"Scanning {len(videos)} videos...")

    results = {}
    for v in videos:
        print(f"Analyzing: {v.name}...", end="", flush=True)
        crop = detect_crop(v, limits=args.limit, round_to=args.round)
        if crop:
            print(f" Found: {crop}")
            results[v.name] = crop
        else:
            print(" No crop found (or full frame).")

    # Save to json if needed (Auto-Boost doesn't strictly need JSON unless integrated)
    # BUT Auto-Boost-Av1an calls this script via subprocess and expects output.
    # Actually, Auto-Boost-Av1an likely imports it OR calls it.
    # The Linux port uses 'tools/dispatch.py' which uses 'tools/cropdetect.py'.
    # I should check 'dispatch.py' usage of this file later.


if __name__ == "__main__":
    main()
