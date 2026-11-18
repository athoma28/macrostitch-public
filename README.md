# Macrostitch

Macrostitch builds a focus-stacked photograph from one or more videos of a still macro subject as focus moves through it. It aligns recorded frames, selects sharp detail, and blends it into a 16-bit TIFF and PNG. The CLI, Python API, and local browser editor use the same pipeline.

## Install

Use macOS or Linux with Python 3.11 or newer. Install FFmpeg and FFprobe and make sure both are on your PATH.

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install .
```

## Stack a video

Place a standard dynamic range BT.709 focus-rack clip at `media/focus-rack.mov`, or substitute your own path. Use a still subject and sweep slowly through the range of focus you want to capture.

```sh
macrostitch stack media/focus-rack.mov \
  --output outputs/stack --config examples/native-spatial.json
```

The [example recipe](examples/native-spatial.json) scans every frame, uses spatial fusion across the whole image, and retains the reference framing. Pass more clip paths before `--output` to combine sweeps of the same scene. Output includes `stack.tiff`, `stack.png`, `preview.jpg`, and `reference.png`.

For a faster first pass, omit `--config` to use proxy selection and pyramid fusion. Use `--progress plain` for terminal progress, `--progress json` for structured events, or `--progress quiet` for the final report alone. The default `auto` chooses based on whether stdout is a terminal.

### Configure a recipe

Copy the example into `recipes/` to customize it. A normalized `roi` gives registration a useful area with recognizable detail. Use `subject_polygon` to outline the region to stack while retaining the reference background. Both use coordinates on the full reference image. With no subject polygon, `subject_only: false` fuses the full frame. Use `canvas_mode: common` to crop to the rectangle valid in every aligned frame.

Native selection accepts `"required_frames": [{"source": 0, "frame": 160}]` to retain a known focal plane. Indices are zero-based and refer to deduplicated clips. Required frames share `max_frames` with the reference and registration anchors, and still undergo alignment and fusion quality checks.

Optional `local_alignment_regions` requires native selection and spatial fusion. Each region supplies normalized `roi`, `fit_roi`, and `validation_rois`, an `anchor` with `source` and `frame`, and pixel settings `max_shift_px` and `feather_px`. Fit and validation rectangles belong inside the region's fully corrected interior where they resolve shared structure. Regions must be disjoint. The pipeline tries a translation of up to six pixels and applies it only when fit and held-out checks improve.

### Edit masks in the browser

From the checkout, start the local service and open [the editor](http://127.0.0.1:8765/).

```sh
macrostitch serve --workspace "$PWD" --port 8765
```

Pick clips and load or upload a recipe. After a render, use the reference preview to draw polygons and choose what each object contributes.

| Control | Result |
| --- | --- |
| **Stack** | Includes the object in focus selection and fusion. |
| **Reference** | Retains the object from the original reference. |
| **Exclude** | Makes the object transparent. |
| **Background opacity** | Sets visibility outside stacked objects and retained context, from 0 to 1. |
| **Surrounding context** | Keeps up to 512 native pixels of reference around the subject. Set opacity below 1 to see the margin. |

Overlaps resolve in this order from highest to lowest priority: Exclude, Reference, then Stack. Recipes support up to 32 named object masks, each with a simple polygon of 3–128 vertices on the normalized reference canvas. Changing clips or the registration ROI can change the framing. Check the new reference before applying saved annotations.

Use **Fit**, **Pan**, and zoom to navigate the reference. Drag vertices in edit mode. **Recent jobs** reopens results and resumes monitoring active jobs. You can download the edited recipe and preview or download the resulting images. The CLI also accepts `--background-opacity`, `--background-margin-px`, and `--object-masks`. The latter takes a JSON array of objects with `name`, `action` (`stack`, `reference`, or `exclude`), and `polygon` fields.

## Python and HTTP

```python
import json
from pathlib import Path

from macrostitch import Config, run_stack

settings = json.loads(Path("examples/native-spatial.json").read_text())
report = run_stack(["media/focus-rack.mov"], "outputs/stack", Config(**settings))
print(report["artifacts"]["stack.tiff"])
```

While the editor service is running, the interactive API docs are at [localhost:8765/docs](http://127.0.0.1:8765/docs). This request runs a stack using the default recipe.

```sh
curl --fail-with-body http://127.0.0.1:8765/jobs \
  --header 'Content-Type: application/json' \
  --data '{"inputs":["media/focus-rack.mov"],"output_dir":"outputs/api-stack"}'
```

Read the returned `id` and poll `GET /jobs/{id}` until `completed`, `failed`, or `interrupted`. On completion, fetch the artifacts listed under `result` through `GET /jobs/{id}/artifacts/{name}`. Paths are scoped to the selected workspace. The service listens on loopback and has no remote authentication layer.

## How it works

1. Fingerprint the inputs, inspect color metadata, and use lower-resolution samples to estimate translation, rotation, and uniform scale. The reference is a recorded frame.
2. In native mode, scan the subject at full resolution and score detail in spatial cells. Select frames that cover reliable structure across the scene.
3. Extract candidate frames as lossless 16-bit TIFFs, refine their alignment, and choose consistent sources for sharp features. Optional local translations align selected surfaces.
4. Blend recorded source pixels in linear light. Composite onto the reference and encode 16-bit output with an sRGB transfer curve.

Parallax, occlusion from defocus, chromatic fringes, and source softness can remain. Weak or uniformly dense texture can fail the native structure checks. The stack preserves processing precision and is limited by the detail the camera recorded.

## Outputs and cache

- `selection.json` records source fingerprints, frame choices, transforms, and coverage. `native-analysis.json` and `fusion.json` describe native scanning and spatial fusion when enabled.
- `source_index.png` labels the dominant source and `confidence.png` measures relative focus evidence. Transparent pixels use source index 65535.
- `reference.png`, `matte.png`, and `subject.png` are available with compositing controls. Outputs use straight alpha when transparency is present.
- `report.json` contains settings, timings, and artifact checksums. `events.json` records stage progress.

The stage cache defaults to `.macrostitch/cache` beside the output directory and uses a 2 GiB retention budget between runs. Use `--cache-max-mib 0` to discard reusable stage entries after a run. Use `--keep-frames` only when you need to inspect the extracted and aligned candidates. These optional exports can use substantial disk space. From the source checkout, `python -m macrostitch.storage --workspace .` previews cleanup of historical scratch artifacts.

Keep your media in `media/`, recipes in `recipes/`, and renders in `outputs/`. Git also ignores the workspace cache and generated `diagnostics/` and `benchmarks/` directories. Generated reports can include absolute paths, input file names, and machine details. Review them before sharing.

## Development and benchmarking

```sh
python -m pip install -e '.[test]'
python -m pytest -q
```

Tests generate synthetic inputs. Use an FFmpeg build with the FFV1 and libx264 encoders and install Node.js to run the editor's JavaScript behavior checks. For the pinned dependency versions in `requirements.lock.txt`, use Python 3.12 or newer. The pins were verified on Python 3.13.

```sh
python -m macrostitch.benchmark media/focus-rack.mov \
  --benchmark-root benchmarks --config examples/native-spatial.json \
  --cold-runs 2 --warm-runs 3 --cache-runs 3
```

Cold runs start from a new application cache. The runner checks repeatability before discarding generated images and arrays. It retains JSON results and logs. Use `python tools/benchmark_native.py --help` to compare warm renders against an existing verified output and populated cache.
