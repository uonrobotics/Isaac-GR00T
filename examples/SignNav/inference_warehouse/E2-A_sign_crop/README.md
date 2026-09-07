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
Instead, the selected `sign_crop` is padded to the same aspect ratio as
`ego_view`, then resized to the ego-view resolution before being fed to GR00T.

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

The web dashboard shows both the selected bbox overlay and the exact `sign_crop`
image being fed into GR00T.
