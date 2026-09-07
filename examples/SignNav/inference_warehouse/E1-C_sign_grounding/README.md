# E1-C sign grounding

SignNav inference for the experiment where the VLM predicts sign bbox
coordinates directly to test whether grounding improves navigation.

Checkpoint root:

`/nas/sujinkim/model/SignNav/gr00t_n1d7/gr00t_n1d7-finetune+sim_v2_lerobot_sign_grounding`

Run:

```bash
./start_gr00t_inference_server.sh
```

The script selects the latest usable nested `checkpoint-*` containing both
`config.json` and `processor_config.json`. Override it with
`MODEL_PATH=/path/to/checkpoint`.

Grounding visualization remains separate:

```bash
uv run python examples/SignNav/visualize_sign_grounding.py --checkpoint "$MODEL_PATH" --image /path/to/image --area 1
```
