#!/usr/bin/env python3
"""
Standalone inference script for MistralMail
"""
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch
import sys

def load_model(model_path="models/mistral-editorial-final"):
    print("Loading MistralMail model...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16
    )
    
    base_model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-v0.1",
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    model = PeftModel.from_pretrained(base_model, model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def generate_response(model, tokenizer, email_text):
    prompt = f"Email: {email_text}\n\nResponse: "
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    inputs = inputs.to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=150,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.15,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("Response:")[-1].strip()

if __name__ == "__main__":
    model, tokenizer = load_model()
    print("Ready! Type your email (or 'quit' to exit):\n")
    
    while True:
        email = input("Email: ")
        if email.lower() == 'quit':
            break
        
        response = generate_response(model, tokenizer, email)
        print(f"\nResponse: {response}\n")
        print("-" * 50)
