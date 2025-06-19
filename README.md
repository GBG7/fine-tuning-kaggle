# Fine-Tuning DeepSeek R1-Distill-Llama-8B (Medical Reasoning)

This project fine-tunes the DeepSeek R1-Distill-Llama-8B model on a medical chain-of-thought dataset using parameter-efficient techniques (LoRA + Unsloth). The goal is to improve step-by-step reasoning performance for clinical Q&A.

---

## 🚀 How to Run

1. Upload the provided `.ipynb` notebook to **Kaggle**.
2. In **Settings > Accelerator**, select **GPU (P100)**.
3. Obtain your API keys for:
    - [Weights & Biases (W&B)](https://wandb.ai/) 
    - [Hugging Face](https://huggingface.co/)
4. On Kaggle, go to **Add-ons > Secrets**:
    - Create a new secret: name = `hugging_face_token`, value = your Hugging Face API token.
    - Create a new secret: name = `wnb_token`, value = your W&B API token.
5. Run the notebook — full setup is automated.

---

## 🧰 Packages & Tools Used

- [`unsloth`](https://github.com/unslothai/unsloth): Efficient fine-tuning and inference for LLMs
  - `FastLanguageModel` to optimize inference & fine-tuning
  - `get_peft_model` to enable LoRA-based fine-tuning
- [`peft`](https://huggingface.co/docs/peft/index): LoRA support for parameter-efficient fine-tuning
- Hugging Face ecosystem:
  - `transformers`: Model handling, tokenization, and training
  - `trl`: Supervised fine-tuning via `SFTTrainer` wrapper
  - `datasets`: Dataset fetching & preprocessing
- `torch`: Deep learning framework (PyTorch backend)
- `wandb`: Real-time experiment tracking and logging

---

## 🩺 Dataset Used

- **Medical O1 Reasoning SFT** — [View on Hugging Face](https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT)  
  This dataset contains medical questions with chain-of-thought reasoning and final answers.

To prevent the model from generating excessively long answers, an EOS (End-of-Sequence) token is appended to each training sample.

---

## 💡 Why LoRA?

Training full LLMs requires updating billions of parameters, which demands significant compute resources.  
**LoRA (Low-Rank Adaptation)** enables efficient fine-tuning by adding small trainable adapters to select layers, while keeping the original model weights frozen.

- Only a fraction of parameters are updated (more than **90% reduction** in trainable size)
- Significantly reduces compute/memory requirements
- Maintains model quality for task-specific adaptation

In this notebook, we applied LoRA using:

```python
model_lora = FastLanguageModel.get_peft_model(model, ...)
