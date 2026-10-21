"""Record an animated GIF of a playground view (default: the speculative race) over CDP.

    python scripts/record_gif.py --view race --seconds 14 --out docs/assets/race.gif

Frames are captured with ``Page.captureScreenshot`` (optionally clipped), deduplicated,
palette-quantised once (shared palette -> small file, no flicker) and written with Pillow.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import shutil
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

from PIL import Image
from screenshots import CDP, CHROME_CANDIDATES  # type: ignore[import-not-found]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8001/")
    ap.add_argument("--view", default="race")
    ap.add_argument("--seconds", type=float, default=14)
    ap.add_argument("--fps", type=float, default=8)
    ap.add_argument("--start-js", default="document.querySelector('.lane.spec .tok:not(.prompt)') !== null",
                    help="JS expression; recording starts when it becomes true")
    ap.add_argument("--clip", default="228,128,1212,700", help="x,y,w,h in CSS pixels")
    ap.add_argument("--width", type=int, default=1440)
    ap.add_argument("--height", type=int, default=900)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--max-width", type=int, default=1100)
    ap.add_argument("--out", type=Path, default=Path("docs/assets/race.gif"))
    ap.add_argument("--port", type=int, default=9334)
    args = ap.parse_args()

    chrome = next(c for c in CHROME_CANDIDATES if Path(c).exists())
    profile = tempfile.mkdtemp(prefix="forge-gif-")
    flags = ["--headless=new", "--disable-gpu", "--hide-scrollbars", f"--user-data-dir={profile}",
             f"--remote-debugging-port={args.port}", "--remote-allow-origins=*"]
    proc = subprocess.Popen([chrome, *flags, "about:blank"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(100):
            try:
                tabs = json.load(urllib.request.urlopen(f"http://127.0.0.1:{args.port}/json"))
                break
            except OSError:
                time.sleep(0.1)
        cdp = CDP(next(t for t in tabs if t["type"] == "page")["webSocketDebuggerUrl"])
        cdp.call("Page.enable")
        cdp.call("Runtime.enable")
        cdp.call("Emulation.setDeviceMetricsOverride", width=args.width, height=args.height,
                 deviceScaleFactor=args.scale, mobile=False)
        cdp.call("Page.navigate", url=f"{args.base}?gif={time.time_ns()}#{args.view}")
        deadline = time.time() + 60
        while time.time() < deadline:
            r = cdp.call("Runtime.evaluate", expression=args.start_js, returnByValue=True)
            if r.get("result", {}).get("value"):
                break
            time.sleep(0.05)
        x, y, w, h = (float(v) for v in args.clip.split(","))
        frames: list[Image.Image] = []
        shown_at: list[float] = []  # capture time at which each *distinct* frame first appeared
        t_end = time.time() + args.seconds
        while time.time() < t_end:
            shot = cdp.call("Page.captureScreenshot", format="png",
                            clip={"x": x, "y": y, "width": w, "height": h, "scale": 1})
            now = time.time()
            img = Image.open(io.BytesIO(base64.b64decode(shot["data"]))).convert("RGB")
            if img.width > args.max_width:
                img = img.resize((args.max_width, round(img.height * args.max_width / img.width)), Image.LANCZOS)
            if not frames or img.tobytes() != frames[-1].tobytes():  # identical frames just extend the hold
                frames.append(img)
                shown_at.append(now)
            time.sleep(max(0.0, 1 / args.fps - (time.time() - now)))
        durations = [int((b - a) * 1000) for a, b in zip(shown_at, shown_at[1:])] + [2500]
        # one shared palette keeps colours stable across frames and the file small
        palette_src = frames[len(frames) // 2].quantize(colors=96, method=Image.MEDIANCUT)
        quant = [f.quantize(palette=palette_src, dither=Image.NONE) for f in frames]
        args.out.parent.mkdir(parents=True, exist_ok=True)
        quant[0].save(args.out, save_all=True, append_images=quant[1:], duration=durations, loop=0, optimize=True)
        print(f"[gif] {args.out}: {len(quant)} frames, {args.out.stat().st_size / 1e6:.1f} MB")
    finally:
        proc.terminate()
        shutil.rmtree(profile, ignore_errors=True)


if __name__ == "__main__":
    main()
