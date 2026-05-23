"""
app.py — Gradio web demo for MT-LNN (Hugging Face Spaces).

Loads a base causal-LM from the Hub (default: Qwen2.5-0.5B-Instruct, supports
Chinese + English) and optionally applies a saved MT-LNN adapter checkpoint.
On free-CPU Spaces the model runs in fp32; on GPU it switches to bfloat16.

Environment variables (set in Space Settings → Variables):
  BASE_MODEL   HF model-id to load  (default: Qwen/Qwen2.5-0.5B-Instruct)
  ADAPTER_PATH local path or HF path to an MT-LNN adapter .pt  (optional)
"""

import os

import torch
import torch.nn.functional as F
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_MODEL   = os.environ.get("BASE_MODEL",   "Qwen/Qwen2.5-0.5B-Instruct")
ADAPTER_PATH = os.environ.get("ADAPTER_PATH", "")
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE        = (torch.bfloat16
                if DEVICE == "cuda" and torch.cuda.is_bf16_supported()
                else torch.float32)

# ---------------------------------------------------------------------------
# Model loading (once at startup)
# ---------------------------------------------------------------------------
print(f"[MT-LNN] Loading {BASE_MODEL} on {DEVICE} ({DTYPE}) …")
_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
if _tokenizer.pad_token is None:
    _tokenizer.pad_token = _tokenizer.eos_token

_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    dtype=DTYPE,
    device_map="auto" if DEVICE == "cuda" else None,
    low_cpu_mem_usage=True,
)

if ADAPTER_PATH and os.path.isfile(ADAPTER_PATH):
    try:
        from mt_lnn.llama_adapter import attach_adapters_from_checkpoint, load_adapter_state
        checkpoint = torch.load(ADAPTER_PATH, map_location="cpu")
        attach_adapters_from_checkpoint(_model, checkpoint)
        load_adapter_state(_model, ADAPTER_PATH, strict=False)
        print(f"[MT-LNN] Adapter loaded from {ADAPTER_PATH}")
    except Exception as exc:
        print(f"[MT-LNN] WARNING: could not load adapter — {exc}")

if DEVICE == "cpu":
    _model = _model.to(DEVICE)
_model.eval()
print("[MT-LNN] Model ready.")

# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------

def _top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 0:
        return logits
    v, _ = torch.topk(logits, min(k, logits.size(-1)))
    return logits.masked_fill(logits < v[:, [-1]], float("-inf"))


def _top_p(logits: torch.Tensor, p: float) -> torch.Tensor:
    if p >= 1.0:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    probs = F.softmax(sorted_logits, dim=-1)
    keep = probs.cumsum(dim=-1) <= p
    keep[..., 0] = True
    mask = torch.zeros_like(logits, dtype=torch.bool)
    mask.scatter_(-1, sorted_idx, keep)
    return logits.masked_fill(~mask, float("-inf"))


def _build_prompt(history: list, message: str) -> str:
    """Build a chat prompt using apply_chat_template when available."""
    messages = []
    for user_msg, bot_msg in history:
        messages.append({"role": "user",      "content": user_msg})
        messages.append({"role": "assistant", "content": bot_msg})
    messages.append({"role": "user", "content": message})

    if getattr(_tokenizer, "chat_template", None):
        return _tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    # Fallback for models without a chat template
    prompt = ""
    for user_msg, bot_msg in history:
        prompt += f"<|user|>\n{user_msg}\n<|assistant|>\n{bot_msg}\n"
    prompt += f"<|user|>\n{message}\n<|assistant|>\n"
    return prompt


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_text(
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
) -> str:
    ids = _tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
    prompt_len = ids.shape[1]
    eos_id = _tokenizer.eos_token_id
    generated_ids = ids.clone()

    with torch.no_grad():
        for _ in range(int(max_new_tokens)):
            out = _model(input_ids=generated_ids)
            logits = out.logits[:, -1, :] / max(float(temperature), 1e-6)
            logits = _top_k(logits, int(top_k))
            logits = _top_p(logits, float(top_p))
            next_id = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            generated_ids = torch.cat([generated_ids, next_id], dim=1)
            if eos_id is not None and next_id.item() == eos_id:
                break

    # Decode only the newly generated tokens to avoid space/encoding issues
    new_tokens = generated_ids[0, prompt_len:]
    return _tokenizer.decode(new_tokens, skip_special_tokens=True)


def chat_stream(
    message: str,
    history: list,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
):
    prompt = _build_prompt(history, message)
    ids = _tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
    prompt_len = ids.shape[1]
    eos_id = _tokenizer.eos_token_id
    generated_ids = ids.clone()

    with torch.no_grad():
        for _ in range(int(max_new_tokens)):
            out = _model(input_ids=generated_ids)
            logits = out.logits[:, -1, :] / max(float(temperature), 1e-6)
            logits = _top_k(logits, int(top_k))
            logits = _top_p(logits, float(top_p))
            next_id = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            generated_ids = torch.cat([generated_ids, next_id], dim=1)

            # Decode ALL new tokens together — fixes SentencePiece space-prefix loss
            new_tokens = generated_ids[0, prompt_len:]
            yield _tokenizer.decode(new_tokens, skip_special_tokens=True)

            if eos_id is not None and next_id.item() == eos_id:
                break


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

_adapter_badge = (
    f"🧠 **MT-LNN adapter active** (`{os.path.basename(ADAPTER_PATH)}`)"
    if ADAPTER_PATH and os.path.isfile(ADAPTER_PATH)
    else "⚙️ Running vanilla base model (no MT-LNN adapter)"
)
_description = f"""
## MT-LNN — Microtubule Linear Neural Network
**Base model:** `{BASE_MODEL}` &nbsp;|&nbsp; **Device:** `{DEVICE}`

{_adapter_badge}

This demo showcases the [MT-LNN architecture](https://huggingface.co/EverestAn/MT-LNN):
a biologically-inspired hybrid that couples a standard transformer with a linear
recurrent network modelling microtubule quantum-coherence dynamics.

支持中英文对话 · Bilingual (Chinese & English) · Type below and hit **Submit**.
"""

with gr.Blocks(title="MT-LNN Demo") as demo:
    gr.Markdown(_description)

    with gr.Tab("💬 Chat"):
        gr.ChatInterface(
            fn=chat_stream,
            additional_inputs=[
                gr.Slider(32, 512, value=200, step=32,   label="Max new tokens"),
                gr.Slider(0.1, 2.0, value=0.7, step=0.05, label="Temperature"),
                gr.Slider(0,   100, value=0,   step=1,   label="Top-k  (0 = off)"),
                gr.Slider(0.0, 1.0, value=0.9, step=0.05, label="Top-p"),
            ],
        )

    with gr.Tab("📝 Completion"):
        prompt_box = gr.Textbox(
            lines=5, placeholder="Enter a prompt… / 输入提示词…", label="Prompt"
        )
        with gr.Row():
            max_tok  = gr.Slider(32,  512,  value=200,  step=32,   label="Max new tokens")
            temp     = gr.Slider(0.1, 2.0,  value=0.7,  step=0.05, label="Temperature")
            top_k_sl = gr.Slider(0,   100,  value=0,    step=1,    label="Top-k  (0 = off)")
            top_p_sl = gr.Slider(0.0, 1.0,  value=0.9,  step=0.05, label="Top-p")
        run_btn = gr.Button("Generate", variant="primary")
        output_box = gr.Textbox(lines=10, label="Generated text", interactive=False)
        run_btn.click(
            fn=generate_text,
            inputs=[prompt_box, max_tok, temp, top_k_sl, top_p_sl],
            outputs=output_box,
        )

    gr.Markdown(
        "---\n"
        "Model weights & code: [EverestAn/MT-LNN](https://huggingface.co/EverestAn/MT-LNN) · "
        "MIT license"
    )

demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    theme=gr.themes.Soft(),
    ssr_mode=False,
)
