from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch

app = Flask(__name__)

print("Loading model...")
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

model = PeftModel.from_pretrained(base_model, "mistral-editorial-final")
tokenizer = AutoTokenizer.from_pretrained("mistral-editorial-final")
tokenizer.pad_token = tokenizer.eos_token
print("Ready!")

@app.route('/')
def home():
    return '''<html><body>
    <h1>Kyle's Email AI</h1>
    <textarea id="e" rows="5" cols="50">Hi Kyle, how's Project Barney?</textarea><br>
    <button onclick="fetch('/gen',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({e:document.getElementById('e').value})}).then(r=>r.json()).then(d=>document.getElementById('o').innerText=d.r)">Generate</button>
    <pre id="o"></pre>
    </body></html>'''

@app.route('/gen', methods=['POST'])
def gen():
    prompt = f"Email: {request.json['e']}\nResponse: "
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(model.device)
    outputs = model.generate(inputs.input_ids, max_new_tokens=100, temperature=0.7, do_sample=True)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).split("Response:")[-1].strip()
    return jsonify({"r": response})

app.run(host='0.0.0.0', port=8081)
