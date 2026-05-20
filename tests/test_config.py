from mindeye3.config import load_config


def test_load_default_config() -> None:
    config = load_config("configs/v0_synthetic.yaml")

    assert config.data.name == "synthetic"
    assert config.data.fmri_dim == 128
    assert config.data.embedding_dim == 64
    assert config.training.device == "cpu"

