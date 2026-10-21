# FLIR Boson Datalogger (Eventide module)

Basic live stream + datalogger for a FLIR Boson, built on
[flirpy](https://github.com/LJMUAstroecology/flirpy). Installs from the
Eventide MODULES tab (repo URL or ZIP).

| Program | What it does |
| ------- | ------------ |
| `boson_datalogger` (`boson_app.py`) | Grabs 16-bit Y16 frames, records segmented MP4 + a timestamp CSV into `<recordings>/boson/`, and publishes raw 16-bit frames on a Unix socket. |
| `boson_mjpeg_server` (`boson_mjpeg_server.py`) | Reads the frame socket, auto-scales to 8-bit, applies a palette, serves MJPEG at `/proxy/flir-boson-datalogger/mjpeg/stream`. |

## Recordings

Each segment is `<UTC>_boson.<ext>` with a matching `<UTC>_boson.csv`
(`frame,timestamp_us` — Pi capture time, µs since epoch). Set with the
`record_format` argument:

| Format | File | Contents | Dashboard |
| ------ | ---- | -------- | --------- |
| `mp4` (default) | `.mp4` | 8-bit, auto-scaled H.264 (fragmented, so segments survive power loss). Same kind of output as the IR camera module. | Listed and playable |
| `ffv1` | `.mkv` | Lossless raw 16-bit Y16 counts (not temperatures). Read back with `ffmpeg -i seg.mkv -f rawvideo -pix_fmt gray16le -` and numpy `<u2`. | **Not listed** unless `"*.mkv"` is added to `RECORDING_EXTENSIONS` in `eventide.py`; playback shows it very dark (8-bit high byte). |

The MP4 scaling is per-frame percentile auto-gain, so absolute temperature
information is not preserved — use `ffv1` when you need the counts.

## Socket wire format

`[u32 len][u64 timestamp_us][u16 width][u16 height][len bytes: row-major u16 LE]`

## Arguments (editable in the MODULES tab)

`segment_duration` (60 s), `record_format` (mp4), `stream_quality` (70), `stream_palette` (gray).
Other CLI flags of `boson_app.py` (`--fps` fallback rate, `--serial-port`,
`--video-device`, `--simulate`) can be added to the manifest command.

## Bench test without a camera

```bash
python3 boson_app.py --simulate --output-dir /tmp/rec --segment-duration 10 &
python3 boson_mjpeg_server.py --port 8080      # http://<pi>:8080/
```
