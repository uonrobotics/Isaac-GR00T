"""Checks for residual configuration shown by the E3-A dashboard."""

import importlib.util
import json
from pathlib import Path
import sys


MODULE = (
    Path(__file__).resolve().parents[2]
    / "examples/SignNav/inference_warehouse/E3-A_sign_state_residual/gr00t_inference_server.py"
)
spec = importlib.util.spec_from_file_location("e3a_inference_server", MODULE)
server = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = server
spec.loader.exec_module(server)


def test_build_model_info_reads_residual_checkpoint_config(tmp_path):
    checkpoint = tmp_path / "state-residual-run" / "checkpoint-100000"
    checkpoint.mkdir(parents=True)
    (checkpoint / "config.json").write_text(
        json.dumps(
            {
                "sign_training_conditioning_mode": "gt_bbox_status",
                "sign_action_residual_scale": 0.1,
            }
        )
    )

    info = server.build_model_info(checkpoint)

    assert info == {
        "model_name": "state-residual-run / checkpoint-100000",
        "model_path": str(checkpoint.resolve()),
        "sign_training_conditioning_mode": "gt_bbox_status",
        "sign_action_residual_scale": 0.1,
    }


def test_dashboard_renders_residual_configuration():
    html = server._DASHBOARD_HTML

    assert 'id="train_condition"' in html
    assert 'id="residual_scale"' in html
    assert "info.sign_training_conditioning_mode" in html
    assert "info.sign_action_residual_scale" in html
    assert server.DEFAULT_SIGN_CONDITIONING_MODE == "gt_bbox_status"
