# E3-A sign state residual

E3-A tests the proposed action-conditioning path. The sign grounding head still
appends its grounded token to the VLM tokens, and the same token is projected
from 2048 to 1536 dimensions and added to the DiT state feature:

```text
state_features_for_DiT = state_features + residual_scale * projector(grounded_token)
```

The residual scale and the training bbox source are loaded from the selected
checkpoint's `config.json`. They are shown in the terminal, live dashboard, and
recorded dashboard video.

## Run the first GT experiment

Train the checkpoint with these settings in
`gr00t/experiment/launch_finetune.py`:

```python
config.model.sign_action_residual_scale = 0.1
config.model.sign_training_conditioning_mode = "gt_bbox_status"
```

Then set the run root or exact checkpoint and start the server:

```bash
cd examples/SignNav/inference_warehouse/E3-A_sign_state_residual
RUN_ROOT=/path/to/training/run ./start_gr00t_inference_server.sh

# Or select one checkpoint directly.
MODEL_PATH=/path/to/checkpoint-10000 ./start_gr00t_inference_server.sh
```

The default inference source is `gt_bbox_status`, matching the first oracle
training experiment, and the default ablation is `normal`, so the grounded token
and state residual remain active.

For the later predicted-box experiment:

```bash
SIGN_CONDITIONING_MODE=pred MODEL_PATH=/path/to/pred-trained-checkpoint \
  ./start_gr00t_inference_server.sh
```

## Comparison order

Use the same dataset, seed, training steps, and simulator episodes for both
checkpoints:

1. GT training + GT inference, residual scale `0.0`.
2. GT training + GT inference, residual scale `0.1`.
3. If step 2 improves action performance, train and evaluate with `pred`.

The first pair isolates whether the state-residual connection helps when bbox
quality is fixed. The third experiment then measures the effect of predicted
grounding errors.

The copied `compare_sign_grounding.py` and `visualize_sign_grounding.py` tools
can still be used for offline grounding analysis. Dashboard videos are saved
under this experiment directory's `recordings/` folder after the client save
prompt is accepted.
