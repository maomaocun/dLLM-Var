from abc import ABC, abstractmethod
class ModelLoader(ABC):
    """
    抽象基类：模型加载器接口。
    所有具体的模型加载器类都应继承本类，并实现 load_model_tokenizer 方法。
    """

    def __init__(self, tokenizer_path, model_path):
        """
        初始化方法，保存分词器路径、模型路径及LoRA开关参数。
        :param tokenizer_path: 分词器文件或目录路径
        :param model_path: 模型文件或目录路径
        :param enable_lora: 是否启用LoRA微调（布尔值）
        """
        self.tokenizer_path = tokenizer_path
        self.model_path = model_path  # 与子类保持一致
        self.tokenizer = None
        self.model = None
    def _load_tokenizer(self):
        """
        私有方法：加载分词器
        子类可重写
        :return: tokenizer对象
        """
        raise NotImplementedError("子类需实现 _load_tokenizer 方法")

    def _load_model(self):
        """
        私有方法：加载模型
        子类可重写
        :return: model对象
        """
        raise NotImplementedError("子类需实现 _load_model 方法")

    @abstractmethod
    def load_model_tokenizer(self):
        """
        抽象方法：加载分词器和模型。
        子类必须实现该方法，返回分词器和模型对象。
        :return: (tokenizer, model)
        """
        pass
