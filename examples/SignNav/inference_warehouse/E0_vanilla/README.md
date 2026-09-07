# E0 vanilla GR00T baseline

Baseline SignNav inference with the vanilla fine-tuned GR00T checkpoint.

Default checkpoint root:

`/nas/sujinkim/model/SignNav/gr00t_n1d7/gr00t_n1d7-finetune+sim_v2_lerobot/gr00t_n1d7-finetune+sim_v2_lerobot--20260824`

Run:

```bash
./start_gr00t_inference_server.sh
```

The script selects the latest usable `checkpoint-*` containing both `config.json`
and `processor_config.json`. Override it with `MODEL_PATH=/path/to/checkpoint`.
