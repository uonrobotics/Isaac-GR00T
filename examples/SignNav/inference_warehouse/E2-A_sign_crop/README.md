# E2-A sign crop + bbox coordinate input

SignNav inference for the experiment that feeds both the RGB ego view and an
online sign crop image, plus bbox coordinate state:

- video: `ego_view`, `sign_crop`
- state: `speed`, `bbox_status`, `bbox_x1`, `bbox_y1`, `bbox_x2`, `bbox_y2`

Default checkpoint root:

`/nas/sujinkim/model/SignNav/gr00t_n1d7/gr00t_n1d7-finetune+sim_v2_lerobot_sign_crop/gr00t_n1d7-finetune+sim_v2_lerobot_sign_crop--20260828`

Run:

```bash
./start_gr00t_inference_server.sh
```

Select the crop source in `start_gr00t_inference_server.sh`:

```bash
SIGN_CROP_SOURCE="${SIGN_CROP_SOURCE:-pred}"  # SAM3 + Qwen3 prediction
# SIGN_CROP_SOURCE="gt"                       # simulator GT bbox
```

With `gt`, Isaac Sim finds USD prims carrying `sign_label`, renders their tight
2D boxes, and sends visible boxes to the inference server. The server selects
the largest visible sign containing the requested Area and uses that same box
for the normalized bbox state and `sign_crop`. If the target sign is not
visible, the crop is black and the bbox state is all zeros. The dashboard draws
the selected GT box in green with `GT BBOX -> CROP`.

The default GT visibility filter follows the positive-bbox lower bounds measured
from this training dataset (307,435 frames): width `0.03125`, height `0.0375`,
and area `0.00125` of the image. It also rejects boxes over 20% occluded or
clipped by the image boundary. Override the thresholds with
`SIGN_GT_MIN_WIDTH_RATIO`, `SIGN_GT_MIN_HEIGHT_RATIO`,
`SIGN_GT_MIN_AREA_RATIO`, and `SIGN_GT_MAX_OCCLUSION` when starting Isaac Sim.

This entrypoint runs an online `SAM3 -> Qwen3-VL` target-marker selector:

1. SAM3 proposes individual sign-panel candidates.
2. Qwen3-VL reads each candidate crop.
3. Python matching selects the panel whose label contains the current target area.
4. The selected panel crop is fed as `sign_crop`.
5. The selected bbox is fed as normalized `bbox_status/x1/y1/x2/y2` state.

The bbox state uses the original `ego_view` coordinate system:

- `bbox_status`: `1` when a target marker is selected, otherwise `0`
- `bbox_x1/y1/x2/y2`: normalized xyxy coordinates in the unpadded ego image

Do not pad `ego_view`, because that would change the bbox coordinate frame.
The selected bbox crop is resized to the same 128×128 square format stored in
the training dataset. `Gr00tN1d7Processor` then center-pads that square to the
`ego_view` aspect ratio before its common image transform. Keeping the online
crop square is necessary to match the training input distribution.

The selector runs in a background thread because SAM3+Qwen3 can be slow. GR00T
keeps using the latest successful crop/bbox until the next selector result is
ready. Before the first result, it uses a black crop and zero bbox state.

SAM3+Qwen3 runs in a separate persistent worker process using
`/home/sujin/workspace/physical-ai/sign_seg_test/sam3_qwen3-test/.venv/bin/python`.
This avoids mixing the GR00T `transformers` install with the newer
`transformers` build required by `Sam3Model` and `Qwen3VLForConditionalGeneration`.
By default the worker uses `--device auto`; pass `--sam3-device cuda` only when
that worker venv's own PyTorch reports CUDA as available.
Worker stderr is written to `.logs/signnav_inference_warehouse/e2a_sam3_qwen3_worker.log`.

The live and recorded dashboards show the checkpoint name/path, crop source,
selected bbox overlay, and the exact `sign_crop` image fed into GR00T.
