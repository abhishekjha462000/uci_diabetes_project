import os
import yaml

def test_params_yaml_structure():
    assert os.path.exists("params.yaml"), "params.yaml is missing"
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    assert "data" in params
    assert "preprocessing" in params
    assert "model" in params
    assert params["preprocessing"]["test_size"] > 0

def test_frontend_exists():
    assert os.path.exists("app/static/index.html"), "Frontend HTML is missing"