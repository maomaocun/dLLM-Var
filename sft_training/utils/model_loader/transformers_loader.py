from transformers import AutoTokenizer, AutoModel
import torch
from .model_loder import ModelLoader

class TransformerModelLoader(ModelLoader):
    def __init__(self, tokenizer_path, model_path):
        super().__init__(tokenizer_path, model_path)
    def _load_tokenizer(self):
        """
        私有方法：加载分词器
        :return: tokenizer对象
        """
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_path,
            padding_side="right",
            trust_remote_code=True,
            use_fast=True
        )
        return self.tokenizer

    def _load_model(self):
        """
        私有方法：加载模型，支持LoRA配置
        :return: model对象
        """
        self.model = AutoModel.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        return self.model

    def load_model_tokenizer(self):
        """
        公开方法：加载分词器和模型
        :return: (tokenizer, model)
        """
        tokenizer = self._load_tokenizer()
        model = self._load_model()
        return tokenizer, model


