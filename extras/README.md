# Auto-Boost-Av1an Extras (Linux)

This folder contains helper scripts for additional workflows.

**Setup:** Make sure to run `chmod +x *.sh` before executing any scripts.

## Video Processing Scripts

| Script | Description |
|--------|-------------|
| `light-denoise.sh` | Applies DFTTest denoise via VapourSynth + x265 lossless encoding |
| `light-denoise-nvidia.sh` | GPU-accelerated denoise using NVEncC (requires NVIDIA GPU) |
| `lossless-intermediary.sh` | Converts video to lossless 10-bit x265 intermediate formats |
| `forced-aspect-remux.sh` | Copies aspect ratio from source files to encoded outputs |
| `compare.sh` | Generates comparison screenshots between video files |

## Audio Processing Scripts

| Script | Description |
|--------|-------------|
| `encode-opus-audio.sh` | Batch audio re-encoding to Opus format |

> **Note:** For standalone audio encoding, see the `audio-encoding/` folder in the root directory which includes AC3, EAC3, and Opus encoders with configurable bitrates.

## Utility Scripts

| Script | Description |
|--------|-------------|
| `disk-usage.sh` | Reports disk usage for tools folders (Linux replacement for NTFS compress) |

## Requirements

- **light-denoise.sh**: VapourSynth, vsdenoise, x265, mkvtoolnix
- **light-denoise-nvidia.sh**: NVEncC (from [rigaya/NVEnc](https://github.com/rigaya/NVEnc)), mkvtoolnix
- **lossless-intermediary.sh**: VapourSynth, x265, mkvtoolnix
- **forced-aspect-remux.sh**: mkvtoolnix
- **compare.sh**: VapourSynth, FFMS2, SubText plugin

## Notes

- `compress-folders` functionality is Windows-only (uses NTFS compression).
- For filesystem-level compression on Linux, consider **btrfs** or **zfs**.
- NVEncC requires an NVIDIA GPU and must be manually installed from GitHub releases.
