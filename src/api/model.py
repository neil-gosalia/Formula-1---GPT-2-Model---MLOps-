from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import os
from dotenv import load_dotenv
from pathlib import Path
load_dotenv()

model = None
tokenizer = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_model():
    global model, tokenizer
    model_path = os.getenv("MODEL_PATH", "neil-gosalia/f1-gpt2")
    if os.path.isdir(model_path):
        model_path = Path(model_path).resolve()

    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    model.to(device)
    model.eval()
    print(f"Model loaded from {model_path}")


def generate_text(prompt: str, top_p: float, temperature: float, max_length: int) -> str:
    if model is None or tokenizer is None:
        raise ValueError("Model not loaded. Call load_model() first")
    input_ids = tokenizer.encode(
        text=prompt,
        return_tensors='pt'
        ).to(device)
    with torch.inference_mode():
        output = model.generate(
            input_ids=input_ids,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(output[0], skip_special_tokens=True)
