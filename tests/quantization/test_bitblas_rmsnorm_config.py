from vllm.model_executor.layers.quantization import get_quantization_config
from vllm.model_executor.layers.quantization.bitblas_rmsnorm import (
    BitBLASRMSNormConfig,
)


def test_get_config_class():
    assert get_quantization_config("bitblas_rmsnorm") is BitBLASRMSNormConfig
