import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2-0.5B-Instruct"

print("Cargando modelo...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
model.eval()
print("Modelo listo.")

LANGUAGES = ["English", "Spanish", "French", "German", "Italian", "Portuguese", "Chinese", "Japanese"]


def translate(text, source_lang, target_lang):
    if not text.strip():
        return ""

    messages = [
        {
            "role": "system",
            "content": (
                f"You are a professional translator. "
                f"Translate the given text from {source_lang} to {target_lang}. "
                f"Output only the translation, nothing else."
            ),
        },
        {"role": "user", "content": text},
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([prompt], return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = outputs[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


with gr.Blocks(title="Traductor — Qwen2-0.5B") as demo:
    gr.Markdown("# Traductor con Qwen2-0.5B\nModelo local, sin API keys.")

    with gr.Row():
        source_lang = gr.Dropdown(choices=LANGUAGES, value="English", label="Idioma origen")
        target_lang = gr.Dropdown(choices=LANGUAGES, value="Spanish", label="Idioma destino")

    input_text = gr.Textbox(label="Texto a traducir", lines=3, placeholder="Escribe aquí...")
    output_text = gr.Textbox(label="Traducción", lines=3, interactive=False)

    translate_btn = gr.Button("Traducir", variant="primary")

    gr.Examples(
        examples=[
            ["I like soccer", "English", "Spanish"],
            ["How are you?", "English", "Spanish"],
            ["What time is it?", "English", "Spanish"],
        ],
        inputs=[input_text, source_lang, target_lang],
    )

    translate_btn.click(fn=translate, inputs=[input_text, source_lang, target_lang], outputs=output_text)

if __name__ == "__main__":
    demo.launch()
