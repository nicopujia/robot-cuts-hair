"""Record a page to an .mp4 with a headless browser (no screen permissions needed).

Frames come from Chrome's DevTools screencast and are encoded with the system ffmpeg.
By default it records the Viser view and crops its side panel; for the web demo use:
    record.py out.mp4 --url "http://127.0.0.1:8123/?demo=1" --until-js "window.__demo?.finished" --no-crop --size 1280 720

Usage (any Python with Playwright; it uses the installed Google Chrome):
    uv run --with playwright python record.py out.mp4 --until-file /path/stop
Recording stops when ``--until-file`` appears (or after ``--max-seconds``).
"""

from __future__ import annotations

import argparse
import base64
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

from playwright.sync_api import sync_playwright

W, H = 1600, 948
PANEL_W, HEADER_H = 356, 48  # Viser side panel and title bar, cropped out of the video


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("--url", default="http://127.0.0.1:8095")
    parser.add_argument("--until-file", type=Path)
    parser.add_argument("--max-seconds", type=float, default=300)
    parser.add_argument("--screenshot", type=Path, help="save one framed still and exit")
    parser.add_argument("--until-js", help="stop when this page expression is truthy")
    parser.add_argument("--no-crop", action="store_true")
    parser.add_argument("--size", type=int, nargs=2, default=[W, H])
    args = parser.parse_args()

    tmp = Path(tempfile.mkdtemp())
    frames: list[tuple[float, Path]] = []
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", args=["--use-angle=metal", "--hide-scrollbars"])
        page = browser.new_page(viewport={"width": args.size[0], "height": args.size[1]})
        page.goto(args.url)
        page.wait_for_selector("canvas", timeout=30_000)
        time.sleep(3)  # haircut.py points the camera via the set_camera RPC
        if args.screenshot:
            page.screenshot(path=str(args.screenshot))
            browser.close()
            return

        # Chrome's DevTools screencast pushes a JPEG for every repaint.
        cdp = page.context.new_cdp_session(page)

        def on_frame(event: dict) -> None:  # type: ignore[type-arg]
            path = tmp / f"{len(frames):06d}.jpg"
            path.write_bytes(base64.b64decode(event["data"]))
            frames.append((event["metadata"]["timestamp"], path))
            cdp.send("Page.screencastFrameAck", {"sessionId": event["sessionId"]})

        cdp.on("Page.screencastFrame", on_frame)
        cdp.send("Page.startScreencast", {"format": "jpeg", "quality": 90})
        start_ts = time.time()
        print("recording", flush=True)
        deadline = time.time() + args.max_seconds
        while time.time() < deadline and not (args.until_file and args.until_file.exists()):
            page.wait_for_timeout(100)  # lets Playwright deliver screencast events
            if args.until_js and page.evaluate(args.until_js):
                break
        end_ts = time.time()  # screencast timestamps are epoch seconds
        cdp.send("Page.stopScreencast")
        browser.close()

    if len(frames) < 2:
        raise SystemExit("no frames were captured")
    # Chrome only sends a frame when the page repaints, so each frame is held
    # until the next one arrived (the last one until recording stopped).
    concat = tmp / "frames.txt"
    lines = []
    for (t, path), t_next in zip(frames, [*(t for t, _ in frames[1:]), end_ts]):
        lines += [f"file '{path}'", f"duration {max(t_next - t, 0.001):.4f}"]
    lines.append(f"file '{frames[-1][1]}'")  # concat demuxer ignores the last duration otherwise
    concat.write_text("\n".join(lines) + "\n")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(concat),
         "-vf", ("" if args.no_crop else f"crop=iw-{PANEL_W}:ih-{HEADER_H}:0:{HEADER_H},")
         + "scale=trunc(iw/2)*2:trunc(ih/2)*2,fps=30",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart",
         str(args.out)],
        check=True,
    )
    span = end_ts - frames[0][0]
    print(f"first frame +{frames[0][0] - start_ts:.2f}s, last +{frames[-1][0] - start_ts:.2f}s, "
          f"stop +{end_ts - start_ts:.2f}s")
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"saved {args.out} from {len(frames)} frames over {span:.1f}s")


if __name__ == "__main__":
    main()
