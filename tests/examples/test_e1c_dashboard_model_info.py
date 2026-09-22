"""Checks for model identity shown by the E1-C dashboard."""

import importlib.util
from pathlib import Path
import sys


MODULE = (
    Path(__file__).resolve().parents[2]
    / "examples/SignNav/inference_warehouse/E1-C_sign_grounding/gr00t_inference_server.py"
)
spec = importlib.util.spec_from_file_location("e1c_inference_server", MODULE)
server = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = server
spec.loader.exec_module(server)


def test_build_model_info_includes_run_and_checkpoint(tmp_path):
    checkpoint = tmp_path / "experiment-w1_5_0p5" / "checkpoint-100000"
    checkpoint.mkdir(parents=True)

    info = server.build_model_info(checkpoint)

    assert info == {
        "model_name": "experiment-w1_5_0p5 / checkpoint-100000",
        "model_path": str(checkpoint.resolve()),
    }


def test_dashboard_renders_model_info_from_info_endpoint():
    html = server._DASHBOARD_HTML

    assert 'id="model_name"' in html
    assert 'id="model_path"' in html
    assert 'fetch("/info")' in html

