import bitsandbytes as bnb
import math
import torch
from torch import nn


class LoraLayer:
    def __init__(
            self,
            in_features: int,
            out_features: int,
    ):
        self.r = {}
        self.lora_alpha = {}
        self.scaling = {}
        self.lora_dropout = nn.ModuleDict({})
        self.lora_A = nn.ModuleDict({})
        self.lora_B = nn.ModuleDict({})
        # For Embedding layer
        self.lora_embedding_A = nn.ParameterDict({})
        self.lora_embedding_B = nn.ParameterDict({})
        # Mark the weight as unmerged
        self.merged = False
        self.disable_adapters = False
        self.in_features = in_features
        self.out_features = out_features

    def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights):
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
        # Actual trainable parameters
        if r > 0:
            self.lora_A.update(nn.ModuleDict({adapter_name: nn.Linear(self.in_features, r, bias=False)}))
            self.lora_B.update(nn.ModuleDict({adapter_name: nn.Linear(r, self.out_features, bias=False)}))
            self.scaling[adapter_name] = lora_alpha / r
        if init_lora_weights:
            self.reset_lora_parameters(adapter_name)
        self.to(self.weight.device)

    def update_layer_embedding(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights):
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
        # Actual trainable parameters
        if r > 0:
            self.lora_embedding_A.update(
                nn.ParameterDict({adapter_name: nn.Parameter(self.weight.new_zeros((r, self.in_features)))})
            )
            self.lora_embedding_B.update(
                nn.ParameterDict({adapter_name: nn.Parameter(self.weight.new_zeros((self.out_features, r)))})
            )
            self.scaling[adapter_name] = lora_alpha / r
        if init_lora_weights:
            self.reset_lora_parameters(adapter_name)
        self.to(self.weight.device)

    def reset_lora_parameters(self, adapter_name):
        if adapter_name in self.lora_A.keys():
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A[adapter_name].weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B[adapter_name].weight)
        if adapter_name in self.lora_embedding_A.keys():
            # initialize a the same way as the default for nn.linear and b to zero
            nn.init.zeros_(self.lora_embedding_A[adapter_name])
            nn.init.normal_(self.lora_embedding_B[adapter_name])


class Linear4bit(bnb.nn.Linear4bit, LoraLayer):
    # Lora implemented in a dense layer
    def __init__(
            self,
            adapter_name,
            in_features,
            out_features,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.0,
            **kwargs,
    ):
        bnb.nn.Linear4bit.__init__(
            self,
            in_features,
            out_features,
            bias=kwargs.get("bias", True),
            compute_dtype=kwargs.get("compute_dtype", torch.float32),
            compress_statistics=kwargs.get("compress_statistics", True),
            quant_type=kwargs.get("quant_type", "nf4"),
        )
        LoraLayer.__init__(self, in_features=in_features, out_features=out_features)

        # Freezing the pre-trained weight matrix
        self.weight.requires_grad = False

        init_lora_weights = kwargs.pop("init_lora_weights", True)
        self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)
        self.active_adapter = adapter_name

    def forward(self, x: torch.Tensor):
        result = super().forward(x)

        if self.disable_adapters or self.active_adapter not in self.lora_A.keys():
            return result
        elif self.r[self.active_adapter] > 0:
            result = result.clone()
            if not torch.is_autocast_enabled():
                expected_dtype = result.dtype
                x = x.to(self.lora_A[self.active_adapter].weight.dtype)
                output = (
                        self.lora_B[self.active_adapter](
                            self.lora_A[self.active_adapter](self.lora_dropout[self.active_adapter](x))
                        ).to(expected_dtype)
                        * self.scaling[self.active_adapter]
                )
            else:
                output = (
                        self.lora_B[self.active_adapter](
                            self.lora_A[self.active_adapter](self.lora_dropout[self.active_adapter](x))
                        )
                        * self.scaling[self.active_adapter]
                )
            result += output
        return result


class Linear8bitLt(bnb.nn.Linear8bitLt, LoraLayer):
    # Lora implemented in a dense layer
    def __init__(
            self,
            adapter_name,
            in_features,
            out_features,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.0,
            **kwargs,
    ):
        bnb.nn.Linear8bitLt.__init__(
            self,
            in_features,
            out_features,
            bias=kwargs.get("bias", True),
            has_fp16_weights=kwargs.get("has_fp16_weights", True),
            memory_efficient_backward=kwargs.get("memory_efficient_backward", False),
            threshold=kwargs.get("threshold", 0.0),
            index=kwargs.get("index", None),
        )
        LoraLayer.__init__(self, in_features=in_features, out_features=out_features)

        # Freezing the pre-trained weight matrix
        self.weight.requires_grad = False
        init_lora_weights = kwargs.pop("init_lora_weights", True)
        self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)
        self.active_adapter = adapter_name

    def forward(self, x: torch.Tensor):
        result = super().forward(x)

        if self.disable_adapters or self.active_adapter not in self.lora_A.keys():
            return result
        elif self.r[self.active_adapter] > 0:
            if not torch.is_autocast_enabled():
                expected_dtype = result.dtype

                if x.dtype != torch.float32:
                    x = x.float()
                output = (
                        self.lora_B[self.active_adapter](
                            self.lora_A[self.active_adapter](self.lora_dropout[self.active_adapter](x))
                        ).to(expected_dtype)
                        * self.scaling[self.active_adapter]
                )
            else:
                output = (
                        self.lora_B[self.active_adapter](
                            self.lora_A[self.active_adapter](self.lora_dropout[self.active_adapter](x))
                        )
                        * self.scaling[self.active_adapter]
                )
            result += output
        return result
