# Fine-Tuning Stable Diffusion for Synthetic Nuclear Fuel Bundle Images

Generating realistic synthetic images of nuclear fuel bundles by fine-tuning **CompVis Stable Diffusion v1.4** on a small, manually collected dataset.

Real images of nuclear fuel bundles are hard to obtain: the assets sit in sensitive environments under strict security protocols. This project builds a text-to-image pipeline that produces clear, varied images of three bundle shapes (cylindrical, rectangular and octagonal) as a secure alternative dataset for engineering and testing.

**Author:** Vinayak Goyal (AI Engineer Intern, Kinetrics Canada) · August 2024

---

## Results at a glance

Each model generated 300 images (100 per shape) and was evaluated against the real dataset with Fréchet Inception Distance (FID, lower is better) and Inception Score (IS, higher is better).

| Training steps | FID ↓ | IS ↑ (mean ± std) |
|---|---|---|
| 1000 | 195.06 | 3.55 ± 0.35 |
| 2000 | 196.43 | **4.15** ± 0.46 |
| **3000** | **185.06** | 4.11 ± 0.56 |
| 5000 | 197.84 | 3.91 ± 0.41 |

**Best model: 3000 steps.** It has the lowest FID while keeping a strong Inception Score. At 5000 steps the model starts overfitting: images become too similar in colour and shape, FID rises and IS drops.

See [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) for the full story of what was done and why.

---

## Repository structure

```
.
├── code/
│   ├── finetune_stable_diffusionv1-4_nuclear_fuel__bundle.ipynb   # data prep, augmentation, training, generation
│   └── nuclear_bundle_images_evaluation.ipynb                     # FID / IS evaluation and plots
├── dataset/
│   └── nuclear_fuel_bundle_dataset/
│       ├── data/          # 302 real images (flat)
│       └── data_sorted/   # same images grouped by shape: cylinder / octagon / rectangular
├── model/
│   └── nuclear_fuel_bundle-model-v1-4_augmented_3000/   # best model: configs, tokenizer, scheduler, logs
├── output_images/
│   ├── 1000 epochs/  2000 epochs/  3000 epochs/  5000 epochs/   # generated images, per shape
├── docs/
│   ├── stable_diffusion_nuclear_fuel_bundle_documentation.docx  # full technical report
│   └── Kinetrics_Internship_Resume_Info.pdf                     # project / role summary
├── INSTRUCTIONS.md      # how to set up, train, generate and evaluate
└── PROJECT_SUMMARY.md   # what was done in the project
```

> **Model weights are not in this repository.** The fine-tuned `.safetensors` files total about 5 GB (the UNet alone is 3.4 GB), well over GitHub's 100 MB per-file limit. The model folder keeps every config, tokenizer and scheduler file, so dropping the weight files back in makes it a complete Diffusers pipeline. See [INSTRUCTIONS.md](INSTRUCTIONS.md#model-weights).

---

## Tech stack

Python 3.10 · PyTorch 2.4 · Hugging Face Diffusers 0.30, Transformers, Datasets 2.20 · Accelerate (fp16 mixed precision) · bitsandbytes 0.43 (8-bit Adam) · TorchMetrics (FID / IS) · Matplotlib · Google Colab (T4 GPU)

## Quick start

```python
import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    "model/nuclear_fuel_bundle-model-v1-4_augmented_3000",  # weights must be present
    torch_dtype=torch.float16,
).to("cuda")

image = pipe("cylinder nuclear fuel bundle").images[0]
image.save("cylinder_bundle.png")
```

Prompts the model was trained on: `cylinder nuclear fuel bundle`, `rectangular nuclear fuel bundle`, `octagon nuclear fuel bundle`.

Full setup, training and evaluation steps are in [INSTRUCTIONS.md](INSTRUCTIONS.md).
