# fine-tuning-kaggle
MEOWWWWWWWWWWWWW -- DeepSeek R1 “Doctor-AI” Fine-Tuning (LoRA + Unsloth) 

# DeepSeek-R1 Medical Fine-Tune 🩺🧠

Fine-tunes **DeepSeek R1-Distill-Llama-8B** for clinical Q & A using a **parameter-efficient** recipe:

* **Unsloth 4-bit quantization** + **LoRA (rank 16)** → fits on a single **NVIDIA P100 (16 GB)**  
* Supervised fine-tuning on 500 chain-of-thought samples from the *Medical O1* dataset  
* **2× fewer irrelevant tokens** and clearer reasoning versus the base model  
* Full run (60 steps) in **< 25 min**; logged to **Weights & Biases**

---

## Quick Start

```python
from unsloth import FastLanguageModel
from transformers import AutoTokenizer

model_id = "your-hf-org/deepseek-r1-medical-lora"
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_id,
    max_seq_length=2048,
    load_in_4bit=True,
)

prompt = "### Question:\nWhat is the first-line therapy for Graves’ ophthalmopathy?\n### Response:\n<think>\n"
FastLanguageModel.for_inference(model)
print(tokenizer.decode(model.generate(tokenizer(prompt, return_tensors="pt").input_ids,
                                     max_new_tokens=256)[0]))
