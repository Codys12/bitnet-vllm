# SPDX-License-Identifier: Apache-2.0
"""BitBLAS quantization with an ephemeral RMSNorm before each linear layer."""

from typing import Any, Optional

import torch

from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.bitblas import (
    BitBLASConfig,
    BitBLASLinearMethod,
)


class BitnetConfig(BitBLASConfig):
    """BitNet quantization config with 2-bit weights and RMSNorm."""

    def __init__(
        self,
        weight_bits: int = 2,
        group_size: Optional[int] = -1,
        desc_act: Optional[bool] = False,
        is_sym: Optional[bool] = False,
        quant_method: Optional[str] = "bitnet",
        lm_head_quantized: bool = False,
    ) -> None:
        if weight_bits != 2:
            raise ValueError(
                "BitnetConfig only supports weight_bits=2")
        super().__init__(weight_bits, group_size, desc_act, is_sym, quant_method,
                         lm_head_quantized)

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "bitnet"

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "BitnetConfig":
        return cls(
            weight_bits=config.get("bits", 2),
            group_size=config.get("group_size", -1),
            desc_act=config.get("desc_act", False),
            is_sym=config.get("sym", False),
            quant_method=config.get("quant_method", "bitnet"),
            lm_head_quantized=config.get("lm_head", False),
        )

    def get_quant_method(self, layer: torch.nn.Module,
                         prefix: str) -> Optional["BitnetLinearMethod"]:
        if isinstance(layer, LinearBase):
            return BitnetLinearMethod(self)
        return None


class BitnetLinearMethod(BitBLASLinearMethod):
    """BitNet linear method with pre RMSNorm."""

    RMS_EPS = 1e-6

    def _apply_rmsnorm(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_f32 = x.to(torch.float32)
        tp_size = get_tensor_model_parallel_world_size()
        if tp_size > 1:
            local_sums = x_f32.pow(2).sum(dim=-1, keepdim=True)
            global_sums = tensor_model_parallel_all_reduce(local_sums)
            variance = global_sums / (tp_size * x_f32.shape[-1])
        else:
            variance = x_f32.pow(2).mean(dim=-1, keepdim=True)
        x_f32 = x_f32 * torch.rsqrt(variance + self.RMS_EPS)
        return x_f32.to(orig_dtype)

    def apply_gptq(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self._apply_rmsnorm(x)
        return super().apply_gptq(layer, x, bias)

    def apply(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return self.apply_gptq(*args, **kwargs)
