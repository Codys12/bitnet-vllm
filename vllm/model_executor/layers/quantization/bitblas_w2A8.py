# SPDX-License-Identifier: Apache-2.0

"""BitBLAS W2A8 (weight-2bit / activation-8bit) weight-only quantization support.

This file defines:
    • BitblasW2A8QuantConfig – QuantizationConfig subclass that registers the
      new method name (``bitblas_w2a8``).
    • BitblasW2A8LinearMethod – LinearMethodBase implementation that:
        ◦ During model loading packs FP16 weights to 2-bit signed (INT2)
          representation (4 values / byte) and stores a per-matrix scale.
        ◦ At runtime performs:
            ▪ Ephemeral RMS-norm on the incoming activation.
            ▪ Dynamic int8 activation quantisation.
            ▪ INT2×INT8→INT32 GEMM via BitBLAS (fallbacks to torch.matmul if
              BitBLAS is unavailable).
            ▪ Result de-quantisation and bias add.

No changes to model code are necessary – the generic Linear layer factory will
instantiate this method whenever ``quant_method == "bitblas_w2a8"`` appears in
``quant_config.json`` or the model's ``config.json``.
"""
from __future__ import annotations

from typing import Any, Optional

import torch
from torch.nn import Parameter, Module

from vllm.logger import init_logger
from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase
from vllm.model_executor.layers.quantization import register_quantization_config, QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig, QuantizeMethodBase
from vllm.model_executor.layers.quantization.ops.bitblas_kernels import matmul_int2_int8
from vllm.model_executor.parameter import PackedvLLMParameter, BasevLLMParameter
from vllm.model_executor.utils import set_weight_attrs

logger = init_logger(__name__)

######################################################################
# Utility helpers                                                   #
######################################################################

def _pack_int2(values: torch.Tensor) -> torch.Tensor:
    """Pack signed 2-bit integers into uint8.

    Expects *values* in range 0-3 (already mapped from signed).
    Groups of 4 values are packed into a single byte (little-endian).
    Shape: [M, K] → [M, ceil(K/4)].
    """
    assert values.dtype in (torch.int8, torch.uint8)
    m, k = values.shape
    pad = (4 - k % 4) % 4
    if pad:
        values = torch.nn.functional.pad(values, (0, pad))
        k += pad
    values_u = values.to(torch.uint8).view(m, -1, 4)
    packed = (values_u[:, :, 0] | (values_u[:, :, 1] << 2) |
              (values_u[:, :, 2] << 4) | (values_u[:, :, 3] << 6))
    return packed.contiguous()

######################################################################
# Quantization config                                               #
######################################################################

@register_quantization_config("bitblas_w2a8")
class BitblasW2A8QuantConfig(QuantizationConfig):
    """Configuration for BitBLAS W2A8 weight-only quantisation."""

    weight_bits: int = 2
    act_bits: int = 8
    eps: float = 1e-6

    def __init__(self):
        super().__init__()

    # ------------------------------------------------------------------
    # Mandatory QuantizationConfig interface implementation
    # ------------------------------------------------------------------

    @classmethod
    def get_name(cls) -> QuantizationMethods:  # type: ignore[override]
        return "bitblas_w2a8"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        # BitBLAS kernels require at least Volta (sm70)
        return 70

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        # Same pattern as other weight-only configs – optional side-car file.
        return ["quant_config.json", "quantize_config.json"]

    @classmethod
    def from_config(cls, _cfg: dict[str, Any]) -> "BitblasW2A8QuantConfig":
        # No tunables beyond method name; ignore file contents.
        return cls()

    # ------------------------------------------------------------------

    def get_quant_method(self, layer: torch.nn.Module, prefix: str
                         ) -> Optional[QuantizeMethodBase]:
        if isinstance(layer, LinearBase):
            return BitblasW2A8LinearMethod(self)
        return None

######################################################################
# Linear method                                                     #
######################################################################

class BitblasW2A8LinearMethod(LinearMethodBase):
    """LinearMethod that performs BitBLAS INT2×INT8 GEMMs."""

    def __init__(self, quant_config: BitblasW2A8QuantConfig):
        self.qconfig = quant_config
        # Will be initialised per-layer in *create_weights*.
        self.bitblas_op = None  # type: ignore

    # --------------------------- weight creation -------------------- #
    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,  # noqa: F841 – unused (shape derivable)
        output_size: int,  # noqa: F841 – unused
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        """Allocate FP16 weight placeholder + quantised buffers."""
        weight_loader = extra_weight_attrs.get("weight_loader")
        output_size_per_partition = sum(output_partition_sizes)

        # FP16 placeholder to receive raw checkpoint shard.
        fp_weight = BasevLLMParameter(
            data=torch.empty(output_size_per_partition,
                               input_size_per_partition,
                               dtype=params_dtype),
            weight_loader=weight_loader,
        )
        layer.register_parameter("_fp_weight", fp_weight)
        set_weight_attrs(fp_weight, extra_weight_attrs)

        # Prepare BitBLAS operator (needed to know packed weight shape).
        try:
            from bitblas import MatmulConfig, Matmul, auto_detect_nvidia_target
            from bitblas.cache import get_database_path, global_operator_cache
        except Exception as err:  # pragma: no-cover
            logger.warning("BitBLAS not available – W2A8 will fall back to torch.matmul. (%s)", err)
            self.bitblas_op = None
        else:
            config = MatmulConfig(
                N=output_size_per_partition,
                K=input_size_per_partition,
                A_dtype="int8",
                W_dtype="int2",
                out_dtype="int32",
                accum_dtype="int32",
                storage_dtype="int8",
                with_scaling=False,  # weight scale supplied as separate arg
                with_zeros=False,
                group_size=-1,
                with_bias=False,
                layout="nt",
            )
            target = auto_detect_nvidia_target()
            cache_path = get_database_path()
            if global_operator_cache.size() == 0:
                global_operator_cache.load_from_database(cache_path, target)
            op = global_operator_cache.get(config)
            if op is None:
                op = Matmul(config, target=target, enable_tuning=False)
                global_operator_cache.add(config, op)
                global_operator_cache.save_into_database(cache_path, target)
            self.bitblas_op = op

        # Shape of packed INT2 weight [N, K//4] after BitBLAS layout (if op exists)
        if self.bitblas_op is not None:
            packed_shape = self.bitblas_op.retrieve_weight_shape()
        else:
            packed_shape = (output_size_per_partition,
                            (input_size_per_partition + 3) // 4)

        qweight = PackedvLLMParameter(
            data=torch.empty(*packed_shape, dtype=torch.uint8, device="cuda"),
            input_dim=1,
            output_dim=0,
            packed_dim=1,
            packed_factor=4,
            bitblas_tile_size=(packed_shape[-2]
                               if self.bitblas_op is not None and self.bitblas_op.propagate_b else None),
            weight_loader=lambda p, lw, *args, **kwargs: None,  # will quantise later
        )
        layer.register_parameter("qweight", qweight)

        # Single scale per matrix.
        w_scale = BasevLLMParameter(
            data=torch.empty(1, dtype=torch.float16, device="cuda"),
            weight_loader=lambda p, lw, *args, **kwargs: None,
        )
        layer.register_parameter("w_scale", w_scale)

        # Store pointer for later processing.
        layer._w2a8_method = self  # type: ignore – circular ref OK

    # ---------------------- post-load quantisation ------------------- #
    def process_weights_after_loading(self, layer: Module) -> None:
        """Quantise the just-loaded FP16 weights into INT2 packed format."""
        if not hasattr(layer, "_fp_weight"):
            raise RuntimeError("FP16 weight placeholder missing – cannot quantise.")

        fp_weight: torch.Tensor = layer._fp_weight.data
        w_scale_val = fp_weight.abs().mean()
        layer.w_scale.data.fill_(w_scale_val)

        # Normalise and clamp to {-1, 0, +1}.
        q_signed = torch.round((fp_weight / w_scale_val).clamp(-1, 1)).to(torch.int8)
        # Map negative to two-s complement domain 0-3.
        q_unsigned = torch.where(q_signed < 0, q_signed + 4, q_signed)
        packed = _pack_int2(q_unsigned.view(fp_weight.shape[0], -1))
        layer.qweight.data.copy_(packed)

        # Free the large FP16 tensor to save memory.
        delattr(layer, "_fp_weight")

    # ----------------------------- GEMM ----------------------------- #
    def apply(self, layer: Module, x: torch.Tensor, bias: Optional[Parameter] = None  # type: ignore[override]
               ) -> torch.Tensor:
        # 1) Ephemeral RMS normalisation (per token / row)
        eps = self.qconfig.eps
        x_norm = x / torch.sqrt(torch.mean(x.float() ** 2, dim=-1, keepdim=True) + eps)

        # 2) Per-token dynamic int8 quantisation.
        xmax = torch.max(torch.abs(x_norm), dim=-1, keepdim=True).values.clamp(min=1e-6)
        x_scale = (127.0 / xmax).to(torch.float32)  # preserve precision for division later.
        x_int8 = torch.round(x_norm * x_scale).to(torch.int8)

        # Flatten batch/seq for GEMM call.
        x_shape_prefix = x_int8.shape[:-1]
        x_2d = x_int8.view(-1, x_int8.shape[-1])

        # 3) INT2×INT8 matmul with BitBLAS (or fallback).
        if self.bitblas_op is not None:
            out_int32 = self.bitblas_op(x_2d, layer.qweight)
        else:
            logger.warning_once(
                "Fallback to torch.matmul for bitblas_w2a8 (BitBLAS missing). Performance will be slow.")
            # Unpack weights rudimentarily for correctness (slow!).
            # Expand packed weight back to int8 0-3 then map to signed values.
            packed = layer.qweight
            unpacked_cols = packed.shape[1] * 4
            unpacked = torch.empty(packed.shape[0], unpacked_cols,
                                   dtype=torch.int8, device=packed.device)
            for shift in range(4):
                unpacked[:, shift::4] = ((packed >> (shift * 2)) & 0x3).to(torch.int8)
            unpacked_signed = torch.where(unpacked == 3, torch.full_like(unpacked, -1), unpacked)
            out_int32 = torch.matmul(x_2d.to(torch.int32), unpacked_signed.to(torch.int32).t())

        # 4) De-quantise.
        out_fp = (out_int32.float() / x_scale.view(-1, 1)) * layer.w_scale

        # 5) Reshape back and add bias if any.
        out = out_fp.view(*x_shape_prefix, out_fp.shape[-1])
        if bias is not None:
            out = out + bias
        return out 