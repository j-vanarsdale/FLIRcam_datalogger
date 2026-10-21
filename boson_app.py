#!/usr/bin/env python3
"""
boson_app.py — FLIR Boson datalogger (Eventide module).

  * Grabs 16-bit (Y16) frames from a FLIR Boson over USB using flirpy.
  * Records them via ffmpeg in fixed-length, UTC-timestamped segments, plus a
    per-segment CSV of capture timestamps. Two formats (--record-format):
      mp4  (default) 8-bit auto-scaled H.264 — shows up in the Eventide
           dashboard's recordings list and PLAYBACK tab, like the IR module.
      ffv1 lossless raw 16-bit Y16 counts in .mkv — for analysis; the
           dashboard does not list .mkv files by default.
  * Publishes every raw frame on a Unix domain socket for boson_mjpeg_server.py
    (or any other consumer).

Socket wire format (per frame):
    [4 bytes  uint32 LE  payload length in bytes]
    [8 bytes  uint64 LE  capture time, microseconds since Unix epoch]
    [2 bytes  uint16 LE  frame width]
    [2 bytes  uint16 LE  frame height]
    [N bytes             row-major uint16 LE pixels (raw Y16 counts)]

Use --simulate to run without a camera (synthetic 640x512 frames).
"""

import argparse
import logging
import os
import queue
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from boson_common import AutoGain, to_8bit

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("boson")

HEADER = struct.Struct("<IQHH")
WRITER_QUEUE_FRAMES = 120      # ~80 MB at 640x512 — absorbs ffmpeg hiccups / segment rolls
MAX_CAPTURE_FAILURES = 30      # consecutive bad grabs before reconnecting
STATUS_INTERVAL_S = 60


# ---------------------------------------------------------------------------
# Frame sources
# ---------------------------------------------------------------------------

class BosonSource:
    """FLIR Boson via flirpy (V4L2 video for frames, serial for housekeeping)."""

    def __init__(self, serial_port=None, video_device=None):
        self.serial_port = serial_port
        self.video_device = video_device
        self.cam = None
        self.fps = 0.0

    def open(self):
        import cv2
        from flirpy.camera.boson import Boson

        # With port=None flirpy auto-detects the serial port by USB VID/PID.
        # Video capture does not need the serial link, so a missing one is fine.
        self.cam = Boson(port=self.serial_port)
        self.cam.setup_video(self.video_device)   # raises if no Boson found
        self.fps = float(self.cam.cap.get(cv2.CAP_PROP_FPS) or 0.0)

        if getattr(self.cam, "conn", None) is not None:
            try:
                log.info("Boson serial=%s firmware=%s part=%s",
                         self.cam.get_camera_serial(),
                         self.cam.get_firmware_revision(),
                         str(self.cam.get_part_number()).strip("\x00 "))
            except Exception as exc:
                log.warning("Could not read camera info over serial: %s", exc)
        else:
            log.warning("Boson serial port not found — housekeeping info unavailable.")

    def grab(self):
        return self.cam.grab()

    def status(self) -> str:
        if getattr(self.cam, "conn", None) is None:
            return ""
        try:
            return f"fpa_temp={self.cam.get_fpa_temperature()}C ffc_state={self.cam.get_ffc_state()}"
        except Exception as exc:
            return f"(serial status failed: {exc})"

    def close(self):
        if self.cam is None:
            return
        for fn in ("release", "close"):
            try:
                getattr(self.cam, fn)()
            except Exception:
                pass
        self.cam = None


class SimSource:
    """Synthetic frames so the pipeline can be exercised without hardware."""

    def __init__(self, fps=30.0, width=640, height=512):
        self.fps, self.w, self.h = fps, width, height
        self._n = 0
        self._next = 0.0
        self._rng = np.random.default_rng(0)

    def open(self):
        self._next = time.monotonic()
        log.info("Simulated Boson %dx%d @ %.0f fps.", self.w, self.h, self.fps)

    def grab(self):
        self._next += 1.0 / self.fps
        delay = self._next - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        y, x = np.mgrid[0:self.h, 0:self.w]
        cx = (self._n * 4) % self.w
        blob = 1500.0 * np.exp(-(((x - cx) ** 2 + (y - self.h / 2) ** 2) / (2 * 40.0 ** 2)))
        noise = self._rng.normal(0, 20, (self.h, self.w))
        self._n += 1
        return (7500 + x * 2 + blob + noise).astype(np.uint16)

    def status(self) -> str:
        return "simulated"

    def close(self):
        pass


# ---------------------------------------------------------------------------
# Unix-socket frame server
# ---------------------------------------------------------------------------

class FrameSocketServer:
    """Broadcasts raw frames to any number of Unix-socket clients.

    A client that can't keep up is dropped (it reconnects on its own) rather
    than being allowed to stall the capture loop — recording comes first.
    """

    SEND_TIMEOUT_S = 0.25

    def __init__(self, path: str):
        self.path = path
        self._clients: list[socket.socket] = []
        self._lock = threading.Lock()
        self._server = None
        self._running = False

    def start(self):
        if os.path.exists(self.path):
            os.unlink(self.path)
        self._server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server.bind(self.path)
        self._server.listen(8)
        self._server.settimeout(1.0)
        self._running = True
        threading.Thread(target=self._accept_loop, daemon=True, name="sock-accept").start()
        log.info("Frame socket listening at '%s'.", self.path)

    def _accept_loop(self):
        while self._running:
            try:
                conn, _ = self._server.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            conn.settimeout(self.SEND_TIMEOUT_S)
            with self._lock:
                self._clients.append(conn)
            log.info("Frame client connected (total: %d).", len(self._clients))

    def send_frame(self, payload: bytes, ts_us: int, width: int, height: int):
        if not self._clients:
            return
        header = HEADER.pack(len(payload), ts_us, width, height)
        with self._lock:
            for client in list(self._clients):
                try:
                    client.sendall(header)
                    client.sendall(payload)
                except OSError:          # includes timeouts and broken pipes
                    self._clients.remove(client)
                    client.close()
                    log.info("Frame client dropped (remaining: %d).", len(self._clients))

    def stop(self):
        self._running = False
        with self._lock:
            for client in self._clients:
                client.close()
            self._clients.clear()
        if self._server:
            self._server.close()
        if os.path.exists(self.path):
            os.unlink(self.path)


# ---------------------------------------------------------------------------
# Segmented writer (ffmpeg)
# ---------------------------------------------------------------------------

class SegmentWriter:
    """Pipes frames to ffmpeg; starts a fresh file every `segment_duration` s.

    mp4:  auto-scaled 8-bit -> libx264 (fragmented MP4, so a segment survives
          power loss). ffv1: raw 16-bit, lossless, intra-only.
    Runs on its own thread behind a queue; if ffmpeg can't keep up, frames are
    dropped and counted instead of blocking the capture loop.
    """

    RETRY_DELAY_S = 5.0

    def __init__(self, output_dir: str, segment_duration: float, record_format: str = "mp4"):
        self.output_dir = output_dir
        self.record_format = record_format
        self._agc = AutoGain()
        self.segment_duration = segment_duration
        self.fps = 30.0                       # container rate; real times are in the CSV
        self.dropped = 0
        self._q: queue.Queue = queue.Queue(maxsize=WRITER_QUEUE_FRAMES)
        self._thread = threading.Thread(target=self._run, daemon=True, name="writer")
        self._proc = None
        self._csv = None
        self._dims = None
        self._frames = 0
        self._seg_start = 0.0
        self._hold_until = 0.0
        os.makedirs(output_dir, exist_ok=True)

    def start(self):
        self._thread.start()

    def put(self, payload: bytes, ts_us: int, width: int, height: int):
        try:
            self._q.put_nowait((payload, ts_us, width, height))
        except queue.Full:
            self.dropped += 1

    def stop(self):
        self._q.put(None)
        self._thread.join(timeout=30)

    # -- internals ---------------------------------------------------------
    def _open_segment(self, width, height):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        base = os.path.join(self.output_dir, f"{stamp}_boson")
        src = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo"]
        if self.record_format == "mp4":
            ext = ".mp4"
            src += ["-pix_fmt", "gray"]
            enc = ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "20", "-g", "60",
                   "-pix_fmt", "yuv420p", "-movflags", "+frag_keyframe+empty_moov"]
        else:
            ext = ".mkv"
            src += ["-pix_fmt", "gray16le"]
            enc = ["-c:v", "ffv1", "-level", "3", "-g", "1", "-slices", "4", "-slicecrc", "1"]
        cmd = src + ["-video_size", f"{width}x{height}", "-framerate", f"{self.fps:g}",
                     "-i", "pipe:0"] + enc + [base + ext]
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        self._csv = open(base + ".csv", "w", buffering=1)
        self._csv.write("frame,timestamp_us\n")
        self._dims = (width, height)
        self._frames = 0
        self._seg_start = time.monotonic()
        log.info("New segment -> '%s%s' (%dx%d @ %.1f fps)", base, ext, width, height, self.fps)

    def _close_segment(self):
        if self._csv:
            self._csv.close()
            self._csv = None
        if self._proc:
            try:
                self._proc.stdin.close()
            except OSError:
                pass
            try:
                self._proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._proc = None
            log.info("Segment closed (%d frames, %d dropped so far).", self._frames, self.dropped)

    def _run(self):
        while True:
            item = self._q.get()
            if item is None:
                break
            payload, ts_us, width, height = item

            if time.monotonic() < self._hold_until:
                self.dropped += 1
                continue

            if self._proc is not None and (
                self._dims != (width, height)
                or self._proc.poll() is not None
                or time.monotonic() - self._seg_start >= self.segment_duration
            ):
                self._close_segment()

            try:
                if self._proc is None:
                    self._open_segment(width, height)
                if self.record_format == "mp4":
                    frame = np.frombuffer(payload, dtype="<u2").reshape(height, width)
                    self._proc.stdin.write(to_8bit(frame, self._agc).tobytes())
                else:
                    self._proc.stdin.write(payload)
                self._csv.write(f"{self._frames},{ts_us}\n")
                self._frames += 1
            except (OSError, ValueError) as exc:
                log.error("Recording error: %s — retrying in %.0f s.", exc, self.RETRY_DELAY_S)
                self._close_segment()
                self._hold_until = time.monotonic() + self.RETRY_DELAY_S
        self._close_segment()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="FLIR Boson datalogger: segmented lossless Y16 recording + raw frame socket.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--output-dir", "-o", default="recordings", metavar="DIR",
                   help="Directory for recorded segments.")
    p.add_argument("--socket", "-s", default="/tmp/boson_frames.sock", metavar="PATH",
                   help="Unix socket path for the raw frame stream.")
    p.add_argument("--segment-duration", "-d", type=float, default=60.0, metavar="SECONDS",
                   help="Length of each recorded segment.")
    p.add_argument("--record-format", choices=("mp4", "ffv1"), default="mp4",
                   help="mp4: 8-bit auto-scaled H.264 (listed/playable in the dashboard). "
                        "ffv1: lossless raw 16-bit .mkv (analysis).")
    p.add_argument("--fps", type=float, default=60.0, metavar="FPS",
                   help="Fallback frame rate for the video container if the camera "
                        "doesn't report one (60 for Boson 640, 9 for export-restricted units).")
    p.add_argument("--serial-port", default=None, metavar="PORT",
                   help="Boson serial port (default: auto-detect).")
    p.add_argument("--video-device", type=int, default=None, metavar="N",
                   help="V4L2 device number, i.e. N in /dev/videoN (default: auto-detect).")
    p.add_argument("--simulate", action="store_true",
                   help="Use synthetic frames instead of a real camera.")
    p.add_argument("--verbose", "-v", action="store_true", help="DEBUG logging.")
    return p.parse_args()


def main():
    args = parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    if args.segment_duration <= 0:
        sys.exit("--segment-duration must be positive.")

    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())

    source = SimSource(fps=args.fps) if args.simulate else BosonSource(args.serial_port, args.video_device)
    server = FrameSocketServer(args.socket)
    writer = SegmentWriter(args.output_dir, args.segment_duration, args.record_format)
    server.start()
    writer.start()

    frames = 0
    last_status = time.monotonic()
    last_count = 0
    try:
        while not stop.is_set():
            # --- (re)connect -------------------------------------------------
            try:
                source.open()
            except Exception as exc:
                log.warning("Camera not available (%s) — retrying in 5 s.", exc)
                source.close()
                stop.wait(5)
                continue
            fps = source.fps if 1.0 <= source.fps <= 200.0 else args.fps
            writer.fps = fps
            log.info("Capturing at %.1f fps.", fps)

            # --- capture loop ------------------------------------------------
            failures = 0
            while not stop.is_set():
                try:
                    img = source.grab()
                except Exception as exc:
                    log.debug("grab() raised: %s", exc)
                    img = None
                ts_us = int(time.time() * 1_000_000)

                if img is None or img.ndim != 2 or img.dtype != np.uint16:
                    if img is not None:
                        log.error("Unexpected frame (shape=%s dtype=%s) — is the camera in Y16 mode?",
                                  img.shape, img.dtype)
                    failures += 1
                    if failures >= MAX_CAPTURE_FAILURES:
                        log.warning("Too many failed grabs — reconnecting.")
                        break
                    time.sleep(0.01)
                    continue
                failures = 0

                height, width = img.shape
                payload = img.astype("<u2", copy=False).tobytes()
                server.send_frame(payload, ts_us, width, height)
                writer.put(payload, ts_us, width, height)
                frames += 1

                now = time.monotonic()
                if now - last_status >= STATUS_INTERVAL_S:
                    log.info("%.1f fps | %d frames | %d dropped | %s",
                             (frames - last_count) / (now - last_status),
                             frames, writer.dropped, source.status())
                    last_status, last_count = now, frames

            source.close()
    finally:
        log.info("Shutting down …")
        source.close()
        writer.stop()
        server.stop()
        log.info("Done.")


if __name__ == "__main__":
    main()
