#!/usr/bin/env python3
"""
boson_mjpeg_server.py — MJPEG live view for boson_app.py.

Reads raw 16-bit frames from boson_app.py's Unix socket, auto-scales them to
8 bits (percentile stretch, smoothed over time), applies a colour palette and
serves a multipart MJPEG stream with aiohttp.

Wire format read from the socket (see boson_app.py):
    [uint32 len][uint64 timestamp_us][uint16 width][uint16 height][len bytes uint16 LE]

Endpoints:
    GET  /                 Small viewer page
    GET  /stream           MJPEG stream (X-Capture-Timestamp on each part)
    GET  /snapshot         Latest frame as one JPEG
    GET  /api/settings     {"quality", "palette", "palettes", "src_width", "src_height"}
    PUT  /api/settings     Update "quality" (1-100) and/or "palette"
    GET  /api/streaming    {"streaming": bool}
    PUT  /api/streaming    {"streaming": bool}
"""

import argparse
import asyncio
import logging
import signal
import socket
import struct
import threading
from typing import Optional

import cv2
import numpy as np
from aiohttp import web

from boson_common import AutoGain, to_8bit

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("boson-mjpeg")

HEADER = struct.Struct("<IQHH")
BOUNDARY = "bosonframe"
IDLE_PROCESS_EVERY = 10     # with no viewers, still refresh /snapshot at 1/10 rate

PALETTES = {
    "gray": None,
    "hot": cv2.COLORMAP_HOT,
    "inferno": cv2.COLORMAP_INFERNO,
    "jet": cv2.COLORMAP_JET,
}

CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, PUT, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
}


@web.middleware
async def cors_middleware(request: web.Request, handler):
    if request.method == "OPTIONS":
        return web.Response(status=204, headers=CORS_HEADERS)
    try:
        response = await handler(request)
    except web.HTTPException as exc:
        exc.headers.update(CORS_HEADERS)
        raise
    response.headers.update(CORS_HEADERS)
    return response


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

class StreamConfig:
    def __init__(self, quality: int, palette: str):
        self._lock = threading.Lock()
        self.quality = quality
        self.palette = palette
        self.src_width = 0       # filled in from incoming frames (read-only to the API)
        self.src_height = 0

    def get(self) -> dict:
        with self._lock:
            return {
                "quality": self.quality,
                "palette": self.palette,
                "palettes": list(PALETTES),
                "src_width": self.src_width,
                "src_height": self.src_height,
            }

    def update(self, **kwargs) -> list[str]:
        errors = []
        with self._lock:
            for key, value in kwargs.items():
                if key == "quality":
                    try:
                        v = int(value)
                    except (TypeError, ValueError):
                        errors.append("quality must be an integer 1-100.")
                        continue
                    if 1 <= v <= 100:
                        self.quality = v
                    else:
                        errors.append("quality must be 1-100.")
                elif key == "palette":
                    if value in PALETTES:
                        self.palette = value
                    else:
                        errors.append(f"palette must be one of {list(PALETTES)}.")
                elif key in ("src_width", "src_height", "palettes"):
                    errors.append(f"'{key}' is read-only.")
                else:
                    errors.append(f"Unknown setting '{key}'.")
        return errors


# ---------------------------------------------------------------------------
# Frame processing (executor thread)
# ---------------------------------------------------------------------------

def process_frame(raw: bytearray, width: int, height: int, quality: int,
                  palette: str, agc: AutoGain) -> bytes:
    frame = np.frombuffer(raw, dtype="<u2").reshape(height, width)
    img8 = to_8bit(frame, agc)
    cmap = PALETTES[palette]
    if cmap is not None:
        img8 = cv2.applyColorMap(img8, cmap)
    ok, buf = cv2.imencode(".jpg", img8, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed.")
    return buf.tobytes()


def recv_exactly(sock: socket.socket, n: int) -> bytearray:
    buf = bytearray(n)
    view = memoryview(buf)
    got = 0
    while got < n:
        r = sock.recv_into(view[got:], n - got)
        if not r:
            raise ConnectionError("Frame socket closed by datalogger.")
        got += r
    return buf


# ---------------------------------------------------------------------------
# Fan-out to HTTP clients
# ---------------------------------------------------------------------------

class FrameDistributor:
    def __init__(self):
        self._queues: list[asyncio.Queue] = []
        self.latest_jpeg: Optional[bytes] = None
        self.streaming = True

    @property
    def client_count(self) -> int:
        return len(self._queues)

    def register(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=2)
        self._queues.append(q)
        return q

    def unregister(self, q: asyncio.Queue):
        if q in self._queues:
            self._queues.remove(q)

    def publish(self, jpeg: bytes, ts_us: int):
        self.latest_jpeg = jpeg
        if not self.streaming:
            return
        for q in self._queues:
            if q.full():
                q.get_nowait()          # drop oldest for slow viewers
            q.put_nowait((jpeg, ts_us))


async def socket_reader(loop, distributor: FrameDistributor, cfg: StreamConfig,
                        socket_path: str, reconnect_delay: float):
    agc = AutoGain()
    while True:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            log.info("Connecting to '%s' …", socket_path)
            await loop.run_in_executor(None, sock.connect, socket_path)
            log.info("Connected to datalogger.")
            n = 0
            while True:
                header = await loop.run_in_executor(None, recv_exactly, sock, HEADER.size)
                length, ts_us, width, height = HEADER.unpack(header)
                raw = await loop.run_in_executor(None, recv_exactly, sock, length)

                if length != width * height * 2:
                    raise ConnectionError(f"Bad frame: {length} bytes for {width}x{height}.")
                cfg.src_width, cfg.src_height = width, height

                n += 1
                idle = distributor.client_count == 0 or not distributor.streaming
                if idle and n % IDLE_PROCESS_EVERY:
                    continue

                params = cfg.get()
                jpeg = await loop.run_in_executor(
                    None, process_frame, raw, width, height,
                    params["quality"], params["palette"], agc,
                )
                distributor.publish(jpeg, ts_us)
        except OSError as exc:      # ConnectionError / FileNotFoundError are OSErrors
            log.warning("Socket error: %s — retrying in %.1f s.", exc, reconnect_delay)
        finally:
            sock.close()
        await asyncio.sleep(reconnect_delay)


# ---------------------------------------------------------------------------
# HTTP handlers
# ---------------------------------------------------------------------------

async def handle_stream(request: web.Request) -> web.StreamResponse:
    distributor: FrameDistributor = request.app["distributor"]
    if not distributor.streaming:
        raise web.HTTPServiceUnavailable(reason="Streaming is currently disabled.")

    response = web.StreamResponse(headers={
        "Content-Type": f"multipart/x-mixed-replace; boundary={BOUNDARY}",
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Connection": "close",
    })
    await response.prepare(request)

    q = distributor.register()
    log.info("Stream started for %s.", request.remote)
    try:
        while True:
            jpeg, ts_us = await asyncio.wait_for(q.get(), timeout=10.0)
            await response.write(
                (f"--{BOUNDARY}\r\nContent-Type: image/jpeg\r\n"
                 f"Content-Length: {len(jpeg)}\r\nX-Capture-Timestamp: {ts_us}\r\n\r\n").encode()
                + jpeg + b"\r\n"
            )
    except (asyncio.TimeoutError, ConnectionResetError, asyncio.CancelledError):
        pass
    finally:
        distributor.unregister(q)
        log.info("Stream ended for %s.", request.remote)
    return response


async def handle_snapshot(request: web.Request) -> web.Response:
    jpeg = request.app["distributor"].latest_jpeg
    if jpeg is None:
        raise web.HTTPServiceUnavailable(reason="No frame available yet.")
    return web.Response(body=jpeg, content_type="image/jpeg", headers={"Cache-Control": "no-cache"})


async def handle_index(request: web.Request) -> web.Response:
    html = """<!DOCTYPE html><html><head><meta charset="UTF-8"><title>Boson Live</title>
<style>body{margin:0;background:#111;color:#aaa;font-family:monospace;display:flex;
flex-direction:column;align-items:center;gap:.75rem;padding:1rem}
img{max-width:100%;border:2px solid #333}a{color:#7dd3fc}</style></head><body>
<img src="stream" alt="Boson live stream">
<div><a href="stream">/stream</a> | <a href="snapshot">/snapshot</a> |
<a href="api/settings">/api/settings</a> | <a href="api/streaming">/api/streaming</a></div>
</body></html>"""
    return web.Response(text=html, content_type="text/html")


async def handle_get_settings(request: web.Request) -> web.Response:
    return web.json_response(request.app["cfg"].get())


async def handle_put_settings(request: web.Request) -> web.Response:
    try:
        body = await request.json()
    except Exception:
        raise web.HTTPBadRequest(reason="Request body must be valid JSON.")
    if not isinstance(body, dict):
        raise web.HTTPBadRequest(reason="JSON body must be an object.")
    cfg: StreamConfig = request.app["cfg"]
    errors = cfg.update(**body)
    if errors:
        return web.json_response({"errors": errors}, status=400)
    log.info("Settings updated: %s", cfg.get())
    return web.json_response(cfg.get())


async def handle_get_streaming(request: web.Request) -> web.Response:
    return web.json_response({"streaming": request.app["distributor"].streaming})


async def handle_put_streaming(request: web.Request) -> web.Response:
    try:
        body = await request.json()
    except Exception:
        raise web.HTTPBadRequest(reason="Request body must be valid JSON.")
    if not isinstance(body, dict) or not isinstance(body.get("streaming"), bool):
        raise web.HTTPBadRequest(reason='Body must be {"streaming": true|false}.')
    distributor: FrameDistributor = request.app["distributor"]
    distributor.streaming = body["streaming"]
    log.info("Streaming %s.", "enabled" if distributor.streaming else "disabled")
    return web.json_response({"streaming": distributor.streaming})


def make_app(distributor: FrameDistributor, cfg: StreamConfig) -> web.Application:
    app = web.Application(middlewares=[cors_middleware])
    app["distributor"] = distributor
    app["cfg"] = cfg
    app.router.add_get("/", handle_index)
    app.router.add_get("/stream", handle_stream)
    app.router.add_get("/snapshot", handle_snapshot)
    app.router.add_get("/api/settings", handle_get_settings)
    app.router.add_put("/api/settings", handle_put_settings)
    app.router.add_get("/api/streaming", handle_get_streaming)
    app.router.add_put("/api/streaming", handle_put_streaming)
    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Boson MJPEG server.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--socket", default="/tmp/boson_frames.sock", metavar="PATH",
                   help="Unix socket produced by boson_app.py.")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--quality", type=int, default=70, metavar="1-100",
                   help="Initial JPEG quality.")
    p.add_argument("--palette", default="gray", choices=list(PALETTES),
                   help="Initial colour palette.")
    p.add_argument("--reconnect-delay", type=float, default=2.0, metavar="SECS")
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args()


async def _main(args: argparse.Namespace):
    cfg = StreamConfig(args.quality, args.palette)
    distributor = FrameDistributor()
    loop = asyncio.get_running_loop()

    reader = asyncio.create_task(
        socket_reader(loop, distributor, cfg, args.socket, args.reconnect_delay))

    runner = web.AppRunner(make_app(distributor, cfg))
    await runner.setup()
    await web.TCPSite(runner, args.host, args.port).start()
    log.info("MJPEG server at http://%s:%d/", args.host, args.port)

    stop = asyncio.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)
    await stop.wait()

    reader.cancel()
    try:
        await reader
    except asyncio.CancelledError:
        pass
    await runner.cleanup()


def main():
    args = parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    if not 1 <= args.quality <= 100:
        raise SystemExit("--quality must be between 1 and 100.")
    asyncio.run(_main(args))


if __name__ == "__main__":
    main()
