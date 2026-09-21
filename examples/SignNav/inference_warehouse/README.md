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

## Dashboard video workflow

Each inference server stages a composed dashboard MP4 for the active episode.
Recording begins with the first dashboard update for that episode.  It stops
after motion has been observed and the measured robot speeds remain below
`|v| < 0.03` and `|w| < 0.03` for 1.5 seconds.  If Isaac Sim cannot report
measured velocity, the recorder falls back to the commanded `vx` and `wz`.

The client starts paused.  In the terminal running
`script/1_start_isaacsim_and_client.sh`, press `s` and Enter to reset to spawn
and start episode 1.  When moving to each later run, press `s` again.  The
client stops the robot and asks once about the completed episode:

```text
[CLIENT] save dashboard video for episode 1? [y/N]:
```

Answer `y` to save the MP4 under the active experiment's `recordings/`
directory, or press Enter / answer `n` to discard the temporary video.  The
spawn reset and next episode start after this decision.  The first episode is
started by the first `s`, so there is no save question before episode 1.

The defaults can be adjusted through the experiment launch scripts because
they forward extra arguments to the server:

```bash
./start_gr00t_inference_server.sh \
  --dashboard-record-fps 5 \
  --dashboard-stop-linear 0.03 \
  --dashboard-stop-angular 0.03 \
  --dashboard-stop-hold-sec 1.5
```

Use `--no-dashboard-video` to disable temporary dashboard recording.
