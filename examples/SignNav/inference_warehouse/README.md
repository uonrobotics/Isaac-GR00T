# SignNav warehouse inference

The warehouse inference runtime is split by experiment:

- `E0_vanilla`: vanilla GR00T SignNav baseline.
- `E1-C_sign_grounding`: checkpoint trained with the sign bbox grounding head.
- `E2-A_sign_crop`: checkpoint that consumes `sign_crop` plus bbox coordinate state.

Each experiment folder owns its own `gr00t_inference_server.py` and launch
script. The old root-level files are left in place for compatibility while the
experiment folders become the clean entrypoints.

Model architecture code still lives in `gr00t/`, but inference code is copied
per experiment so checkpoint paths, input adapters, and launch behavior can
evolve independently.
