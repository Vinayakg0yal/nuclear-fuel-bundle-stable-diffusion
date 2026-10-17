# Project Summary — What I Did

**Project:** Fine-Tuning Stable Diffusion for Synthetic Generation of Nuclear Fuel Bundle Images
**Role:** AI Engineer Intern, Kinetrics Canada
**Completed:** August 2024

## The problem

Images of nuclear fuel bundles are scarce. The bundles sit in sensitive, access-controlled environments, so photographing them is slow and restricted. Engineering and testing work needed more images than could realistically be captured. On top of that, bundle designs vary a lot, and a model trained on so little data risks overfitting and producing near-identical images.

## The goal

Fine-tune a text-to-image model that produces clear, realistic and *varied* images of nuclear fuel bundles in each of the main shapes, without overfitting.

## What I did, step by step

### 1. Built and organised the dataset
- Worked with a manually collected set of **307 real images**.
- Cleaned and resized them, then sorted them into three shape groups: **144 cylinder, 77 rectangular, 55 octagon**.
- Applied light augmentation (horizontal flip, brightness jitter, random resized crop to 256×256).
- Wrote a `metadata.csv` pairing every image with a text caption, and published the dataset to the Hugging Face Hub for training.

### 2. Ran baseline training
- Fine-tuned **Stable Diffusion v1.2 and v1.4** (CompVis, Hugging Face) for 1000 and 2000 steps, with every image captioned simply as "nuclear fuel bundle".
- **Result:** chaotic, unstructured images. The models blended features from different bundle types because nothing told them the shapes apart.

### 3. Re-labelled the data by shape
- Rewrote every caption to include the shape: "cylinder / rectangular / octagon nuclear fuel bundle".
- This gave the model a way to learn each shape as its own concept, and became the key improvement in the project.

### 4. Focused training on SD v1.4
- Retrained v1.4 on the re-labelled data at **1000, 2000, 3000 and 5000 steps**.
- Made it fit on a free Colab **T4 GPU** with 8-bit Adam (bitsandbytes), fp16 mixed precision (Accelerate), gradient checkpointing, gradient accumulation and EMA weights.
- Observations:
  - **2000 steps:** cylinders looked good, rectangles improved, octagons still messy — matching how many examples of each shape were in the data.
  - **3000 steps:** clear, distinct images across all three shapes.
  - **5000 steps:** overtraining — images became very similar in colour and shape, so I stopped there.

### 5. Evaluated the models quantitatively
- Generated **100 images per shape for every model** (300 per model).
- Scored each model against the real images with **FID** and **Inception Score** using TorchMetrics, and plotted both against training length.

| Steps | FID ↓ | IS ↑ |
|---|---|---|
| 1000 | 195.06 | 3.55 ± 0.35 |
| 2000 | 196.43 | 4.15 ± 0.46 |
| **3000** | **185.06** | 4.11 ± 0.56 |
| 5000 | 197.84 | 3.91 ± 0.41 |

### 6. Picked the best model and documented the work
- **3000 steps** was the sweet spot: lowest FID (closest to real images) with a strong IS (good quality and diversity).
- Wrote the technical report (`docs/stable_diffusion_nuclear_fuel_bundle_documentation.docx`) covering the problem, method, results and conclusions.

## Key takeaways
- **Captions matter more than training time.** Shape-specific labels fixed the blended-feature problem that longer training could not.
- **More training is not always better.** Watching FID/IS across checkpoints showed where quality peaked and where overfitting began.
- **Class balance shows up in the output.** The shape with the fewest images (octagon) was the last to come out clean.
- **Diffusion models can fill data gaps in restricted domains**, giving a secure synthetic alternative when real data is hard to collect.

## Skills and tools
Python · PyTorch · Hugging Face Diffusers, Transformers, Datasets · Accelerate · bitsandbytes · TorchMetrics / torch-fidelity · Matplotlib · Google Colab · Linux · Git
