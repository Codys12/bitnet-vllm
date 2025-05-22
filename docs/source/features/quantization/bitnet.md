(bitnet)=

# BitNet

BitNet builds upon BitBLAS by inserting an RMSNorm layer before each linear layer and packing weights as 2 bits each in `int8` tensors. This format enables extremely compact checkpoints while retaining accuracy.

```console
pip install bitblas>=0.1.0
```

To enable detailed debugging during model loading set the environment variable:

```console
export VLLM_LOGGING_LEVEL=DEBUG
```

## Load a BitNet checkpoint

```python
from vllm import LLM
import torch

model_id = "path/to/bitnet/model"
llm = LLM(model=model_id, dtype=torch.float16, trust_remote_code=True, quantization="bitnet")
```

