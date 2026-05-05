from flask import Flask, request, jsonify, render_template
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

app = Flask(__name__)

print("Cargando modelo...")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct", dtype=torch.float32)
model.eval()
print("Modelo listo.")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/translate", methods=["POST"])
def translate():
    data = request.json
    text = data["text"]
    source = data["source"]
    target = data["target"]

    messages = [
        {"role": "system", "content": f"You are a translator. Translate from {source} to {target}. Output only the translation."},
        {"role": "user", "content": text},
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([prompt], return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=100, pad_token_id=tokenizer.eos_token_id)

    result = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return jsonify({"translation": result.strip()})


if __name__ == "__main__":
    app.run(debug=False)
