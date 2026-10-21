"""Capture README screenshots of the playground with headless Chrome over the DevTools protocol.

Unlike ``chrome --screenshot``, this waits in *real* time (so SSE streams finish), can move
the mouse to trigger hover states, and reports console errors.

    python scripts/screenshots.py                      # all views -> docs/assets/
    python scripts/screenshots.py --views generate --wait 6
"""

from __future__ import annotations

import argparse
import base64
import json
import shutil
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

import websocket  # websocket-client

CHROME_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    "/usr/bin/google-chrome", "/usr/bin/chromium", "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
]

# view -> (wait seconds, list of actions). Actions: ("move", x, y) | ("click", x, y) | ("js", code) | ("sleep", s)
PLAN: dict[str, tuple[float, list]] = {
    "generate": (9, [("js", "document.querySelectorAll('.tok:not(.prompt)')[18]"
                            "?.dispatchEvent(new MouseEvent('click'))")]),
    "race": (12, []),
    "attention": (6, []),
    "embeddings": (6, []),
    "tokenizer": (3, [("js", "document.querySelectorAll('#view-tokenizer .seg.text button')[1]?.click()"),
                      ("sleep", 1.5)]),
    "training": (4, []),
}


class CDP:
    def __init__(self, ws_url: str):
        self.ws = websocket.create_connection(ws_url, timeout=60, suppress_origin=True)
        self.i = 0
        self.console: list[str] = []

    def call(self, method: str, **params):
        self.i += 1
        self.ws.send(json.dumps({"id": self.i, "method": method, "params": params}))
        while True:
            msg = json.loads(self.ws.recv())
            if msg.get("method") == "Runtime.exceptionThrown":
                self.console.append(msg["params"]["exceptionDetails"].get("text", "exception"))
            elif msg.get("method") == "Runtime.consoleAPICalled" and msg["params"]["type"] in ("error", "warning"):
                self.console.append(" ".join(str(a.get("value", "")) for a in msg["params"]["args"]))
            if msg.get("id") == self.i:
                if "error" in msg:
                    raise RuntimeError(msg["error"])
                return msg.get("result", {})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8001/")
    ap.add_argument("--out", type=Path, default=Path("docs/assets"))
    ap.add_argument("--views", nargs="*", default=list(PLAN))
    ap.add_argument("--width", type=int, default=1440)
    ap.add_argument("--height", type=int, default=900)
    ap.add_argument("--scale", type=float, default=2.0, help="device scale factor (2 = retina)")
    ap.add_argument("--wait", type=float, default=None, help="override per-view wait")
    ap.add_argument("--port", type=int, default=9333)
    args = ap.parse_args()

    found = [c for c in CHROME_CANDIDATES if Path(c).exists()]
    chrome = found[0] if found else shutil.which("chrome") or shutil.which("chromium")
    if not chrome:
        raise SystemExit("Chrome/Chromium not found")
    profile = tempfile.mkdtemp(prefix="forge-shot-")
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
        page = next(t for t in tabs if t["type"] == "page")
        cdp = CDP(page["webSocketDebuggerUrl"])
        cdp.call("Page.enable")
        cdp.call("Runtime.enable")
        cdp.call("Network.enable")
        cdp.call("Emulation.setDeviceMetricsOverride", width=args.width, height=args.height,
                 deviceScaleFactor=args.scale, mobile=False)
        args.out.mkdir(parents=True, exist_ok=True)
        for view in args.views:
            wait, actions = PLAN.get(view, (5, []))
            cdp.call("Network.setCacheDisabled", cacheDisabled=True)
            cdp.call("Page.navigate", url=f"{args.base}?shot={time.time_ns()}#{view}")
            time.sleep(args.wait if args.wait is not None else wait)
            for act in actions:
                if act[0] == "js":
                    cdp.call("Runtime.evaluate", expression=act[1])
                elif act[0] in ("move", "click"):
                    cdp.call("Input.dispatchMouseEvent", type="mouseMoved", x=act[1], y=act[2])
                    if act[0] == "click":
                        for t in ("mousePressed", "mouseReleased"):
                            cdp.call("Input.dispatchMouseEvent", type=t, x=act[1], y=act[2], button="left",
                                     clickCount=1)
                elif act[0] == "sleep":
                    time.sleep(act[1])
            time.sleep(0.6)
            shot = cdp.call("Page.captureScreenshot", format="png")
            path = args.out / f"{'tokenizer_view' if view == 'tokenizer' else view}.png"
            path.write_bytes(base64.b64decode(shot["data"]))
            print(f"[shot] {path}")
        if cdp.console:
            print("[shot] console messages:", *cdp.console[:10], sep="\n  ")
    finally:
        proc.terminate()
        try:
            proc.wait(5)
        except subprocess.TimeoutExpired:
            proc.kill()
        shutil.rmtree(profile, ignore_errors=True)


if __name__ == "__main__":
    main()
