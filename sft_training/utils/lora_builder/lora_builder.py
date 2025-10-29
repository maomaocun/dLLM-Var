from peft import LoraConfig, get_peft_model, TaskType
import torch
class LoraBuilder:
    def __init__(self,lora_config:dict):
        self.lora_config = vars(lora_config)
    def get_Lora(self, model):
        config = LoraConfig(
            r=self.lora_config.get("rank", 32),
            lora_alpha=self.lora_config.get("lora_alpha", 256),
            target_modules=self.lora_config.get("target_modules", ["q_proj", "k_proj", "v_proj"]),
            lora_dropout=self.lora_config.get("lora_dropout", 0.05),
            bias=self.lora_config.get("bias", "none"),
            task_type=self.lora_config.get("task_type", TaskType.CAUSAL_LM),
        )
        model = get_peft_model(model, config).to(torch.bfloat16)
        return model