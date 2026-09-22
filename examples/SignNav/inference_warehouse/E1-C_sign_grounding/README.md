# E1-C sign grounding

SignNav inference for the experiment where the VLM predicts sign bbox
coordinates directly to test whether grounding improves navigation.

Checkpoint root:

`/nas/sujinkim/model/SignNav/gr00t_n1d7/gr00t_n1d7-finetune+sim_v2_lerobot_sign_grounding`

Run:

```bash
./start_gr00t_inference_server.sh
```

GT bbox action-conditioning experiment:

```bash
# Baseline: predicted bbox and predicted status gate
SIGN_CONDITIONING_MODE=pred ./start_gr00t_inference_server.sh

# Simulator GT coordinates, predicted status gate
SIGN_CONDITIONING_MODE=gt_bbox ./start_gr00t_inference_server.sh

# Simulator GT coordinates and GT found/not-found gate (oracle run)
SIGN_CONDITIONING_MODE=gt_bbox_status ./start_gr00t_inference_server.sh
```

The Isaac Sim observation server registers warehouse prims carrying a
`sign_label` attribute and returns their rendered tight 2D boxes. The client
forwards those boxes, and the E1-C server selects the largest visible panel that
contains the current target area. It converts pixel `xyxy` into normalized
`cxcywh` before policy preprocessing. The dashboard draws predicted boxes in red
and simulator GT boxes in green. In a GT mode the green overlay reads
`GT BBOX -> ACTION`, and the `Action BBox` row reports the active mode.

The script selects the latest usable nested `checkpoint-*` containing both
`config.json` and `processor_config.json`. Override it with
`MODEL_PATH=/path/to/checkpoint`.

Grounding visualization remains separate:

```bash
uv run python examples/SignNav/visualize_sign_grounding.py --checkpoint "$MODEL_PATH" --image /path/to/image --area 1
```

### Training conditioning source

Training uses predicted bbox/status for the grounded token by default. Select
the source directly in `gr00t/experiment/launch_finetune.py`, keeping exactly
one assignment active:

```python
config.model.sign_training_conditioning_mode = "pred"
# config.model.sign_training_conditioning_mode = "gt_bbox_status"
```

The selected value is saved as `sign_training_conditioning_mode` in the
checkpoint `config.json`. This setting controls training only; inference still
uses `--sign-conditioning-mode` independently.

Compare two dataset-mode grounding results (run from the repository root):

```bash
uv run python examples/SignNav/inference_warehouse/E1-C_sign_grounding/compare_sign_grounding.py \
  examples/SignNav/inference_warehouse/E1-C_sign_grounding/visualization_sign_head_test_1_5_0p5 \
  examples/SignNav/inference_warehouse/E1-C_sign_grounding/visualization_sign_head_test_1_5_0p5+bbox_detached \
  --name-a baseline --name-b bbox_detached
```

Each argument accepts a result directory containing exactly one `summary*.json`,
or a summary JSON file. The script reads GT from the summary's LeRobot dataset;
use `--dataset /path/to/dataset` if it has moved. Requires numpy, pandas and a
parquet engine; no GPU or model loading is needed. Both results must have matching
episode/frame sets, prompts, goals, query indices and model input keys.

The terminal report prints status metrics, bbox IoU/L1/GIoU, per-goal and size
breakdowns, and the largest
per-frame improvements/regressions (`--top 5`). Bbox metrics use all GT-found
frames, including missed detections. Reported losses are unweighted.

The report highlights detection success (found and IoU >= 0.5), mean IoU, false
positives and misses first. Rates use percentages and percentage-point changes.
Goal/size groups are sorted by absolute IoU change, and a leave-one-out summary
shows whether the largest-changing sample drives the average difference.
Improvements are green and regressions red on a terminal; redirected output stays
plain. Use `--color always|never` to override (`NO_COLOR` disables automatic color).
Use `--details` to also print source paths and status confusion matrices.
Use `--explain` to print metric definitions in Korean; default output contains only
tables, comparison statistics and evaluation scope.

### Interactive comparison

Add `--web` to the same command to open a local comparison server:

```bash
uv run python examples/SignNav/inference_warehouse/E1-C_sign_grounding/compare_sign_grounding.py \
  examples/SignNav/inference_warehouse/E1-C_sign_grounding/visualization_sign_head_test_1_5_0p5 \
  examples/SignNav/inference_warehouse/E1-C_sign_grounding/visualization_sign_head_test_1_5_0p5+bbox_detached \
  --name-a baseline --name-b bbox_detached --web
```

Open `http://127.0.0.1:7865`. The left panel contains the report and clickable
improvement/regression samples. The right panel shows paired A/B images with
per-sample IoU, L1, status and found probability. Navigate with previous/next,
left/right arrow keys, sample search or the sample selector. Filters select IoU
improvements, regressions, status changes or GT-found samples with B IoU below 0.5.
Zoom and scrolling are synchronized between the two images. Original images can
also be opened in separate tabs. No additional web framework is required.

Use `--port` to change the port and `--host` to change the bind address (default:
localhost). For a remote SSH workspace, forward port 7865 through VS Code's Ports
panel or `ssh -L 7865:127.0.0.1:7865 USER@HOST`, then open the same local URL.
Stop the server with Ctrl+C. Omit `--web` to retain the terminal report.
