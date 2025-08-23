import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
import threading

class VLMService:
    """Singleton service for running SmolVLM video description"""
    _instance = None
    _lock = threading.Lock()
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            with self._lock:
                if not self._initialized:
                    self.model_name = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"
                    self.device = "cuda" if torch.cuda.is_available() else "cpu"
                    print("Initializing SmolVLM model...", self.device)
                    self.processor = AutoProcessor.from_pretrained(self.model_name)
                    self.model = AutoModelForImageTextToText.from_pretrained(self.model_name).to(self.device)
                    print("SmolVLM model initialized!")
                    self._initialized = True

    def describe_video(self, video_path: str) -> str:
        image_path = './data/test.png'
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant that can understand images."}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "path": image_path},
                    {"type": "text", "text": "Describe this image in detail"}
                ]
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.device)
        outputs = self.model.generate(**inputs, max_new_tokens=512)
        description = self.processor.decode(outputs[0], skip_special_tokens=True)
        return description

# Global instance
vlm_service = VLMService()
