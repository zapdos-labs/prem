import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

# Load model and processor
model_name = "HuggingFaceTB/SmolVLM2-256M-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained(model_name)
model = AutoModelForImageTextToText.from_pretrained(model_name).to(device)

# Input video path
video_path = "./data/test.mp4"

# Prepare the "chat" input
messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "You are a helpful assistant that can understand videos."}]
    },
    {
        "role": "user",
        "content": [
            {"type": "video", "path": video_path},
            {"type": "text", "text": "Describe this video. Be specific about the content type and activities."}
        ]
    }
]

# Convert to model inputs
inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt"
).to(device)

# Generate output
outputs = model.generate(**inputs, max_new_tokens=512)
description = processor.decode(outputs[0], skip_special_tokens=True)

# Clean up and print
description_text = description.split("assistant: ")[-1]
print("Video description:\n", description_text)
