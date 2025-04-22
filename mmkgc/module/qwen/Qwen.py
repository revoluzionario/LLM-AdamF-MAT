import torch
import torch.nn as nn
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig
)
from peft import (
    prepare_model_for_kbit_training,
    get_peft_model,
    LoraConfig,
    TaskType
)
from ..BaseModule import BaseModule

class Qwen2_5_VL_Peft(BaseModule):
    """
    Wrapper around Qwen2.5-VL that:
      - Loads/quantizes the backbone
      - Applies PEFT (LoRA)
      - Exposes methods for encoding, hidden state retrieval, and optimizer preparation
    """
    def __init__(
        self,
        base_ckpt="Qwen/Qwen2.5-VL-7B-Instruct",
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        quantize=True,
        device="cuda"
    ):
        super(Qwen2_5_VL_Peft, self).__init__()
        self.device = torch.device(device)
        # 1) Optionally quantize to 4-bit for QLoRA
        if quantize:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
            backbone = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                base_ckpt,
                quantization_config=bnb_config,
                device_map="auto",
                output_hidden_states=True
            )
        else:
            backbone = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                base_ckpt,
                device_map={"": device},
                output_hidden_states=True
            )
        # 2) Prepare for k-bit training (freezes norms, casts etc.)
        backbone = prepare_model_for_kbit_training(backbone)
        # 3) Apply LoRA adapters
        lora_cfg = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            inference_mode=False,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "dense", "dense_1"]
        )
        self.model = get_peft_model(backbone, lora_cfg)
        self.model.to(self.device)
        # 4) Processor for tokenization & vision features
        self.processor = AutoProcessor.from_pretrained(base_ckpt)

    def encode(
        self,
        images: list,
        texts: list,
        es: torch.Tensor,
        max_length:int=128
    ) -> torch.Tensor:
        """
        Encode a batch of (image, text, structural_embs) pairs into a joint embedding.
        Returns: tensor of shape (B, hidden_size)
        """
        inputs = self.processor(
            images=images,
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        ).to(self.device)
        B = es.size(0)
        es = es.to(self.device)               # (B, de)
        input_ids      = inputs.input_ids       # (B, S)
        attention_mask = inputs.attention_mask  # (B, S)

        # 3) Build inputs_embeds by prepending es as the soft token
        token_embeds   = self.qwen_model.get_input_embeddings()(input_ids)  # (B, S, de)
        es_unsq        = es.unsqueeze(1)                                    # (B, 1, de)
        inputs_embeds  = torch.cat([es_unsq, token_embeds], dim=1)          # (B, 1+S, de)

        # 4) Adjust attention mask
        extra_mask    = torch.ones((B, 1), device=self.device, dtype=attention_mask.dtype)
        new_attn_mask = torch.cat([extra_mask, attention_mask], dim=1)      # (B, 1+S)

        with torch.no_grad():
            outputs = self.model.base_model(
                pixel_values=inputs.pixel_values,
                input_ids=inputs_embeds,
                attention_mask=new_attn_mask,
                output_hidden_states=True,
                return_dict=True
            )
        # Pool the [CLS]‐like token
        return outputs.hidden_states[-1][:, 0, :]

    def get_hidden_states(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        pixel_values: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Return the last hidden states for arbitrary token inputs.
        Args:
          - input_ids, attention_mask: from tokenizer/processor
          - pixel_values: optional image tensor (B,3,224,224)
        Returns: (B, seq_len, hidden_size)
        """
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if pixel_values is not None:
            inputs["pixel_values"] = pixel_values.to(self.device)
        outputs = self.model.base_model(
            **{k: v.to(self.device) for k, v in inputs.items()},
            output_hidden_states=True,
            return_dict=True
        )
        return outputs.hidden_states[-1]

    def get_trainable_parameters(self):
        """
        Returns an iterator over parameters with requires_grad=True (LoRA adapters).
        """
        return filter(lambda p: p.requires_grad, self.model.parameters())

    def get_optimizer(self, lr=5e-5, weight_decay=0.0):
        """
        Construct an AdamW optimizer over the PEFT‑enabled parameters.
        """
        from torch.optim import AdamW
        return AdamW(self.get_trainable_parameters(), lr=lr, weight_decay=weight_decay)