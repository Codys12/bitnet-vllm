from vllm.model_executor.layers.quantization import get_quantization_config
from vllm.model_executor.layers.quantization.bitnet import (
    BitnetConfig,
)


def test_get_config_class():
    assert get_quantization_config("bitnet") is BitnetConfig
