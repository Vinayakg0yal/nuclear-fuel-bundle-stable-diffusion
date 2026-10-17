# Instructions

How to reproduce the project: prepare the dataset, fine-tune Stable Diffusion v1.4, generate images and evaluate them. The notebooks were written for **Google Colab with a T4 GPU**; the same steps work on any Linux machine with an NVIDIA GPU (about 15 GB of VRAM is enough with the memory settings below).

---

## 1. Environment

Versions used in the original runs:

| Package | Version |
|---|---|
| Python | 3.10.12 |
| PyTorch | 2.4.0 |
| Diffusers | 0.30.0 |
| Datasets | 2.20.0 |
| bitsandbytes | 0.43.3 |

Install (this is what the first cell of the training notebook does):

```bash
pip install datasets
git clone https://github.com/huggingface/diffusers.git
pip install ./diffusers
pip install -U -r diffusers/examples/text_to_image/requirements.txt
pip install bitsandbytes
pip install "torchmetrics[image]"   # for evaluation
```

You also need a Hugging Face account (to push the dataset and download the base model):

```bash
huggingface-cli login
```

## 2. Prepare the dataset

The real images are in `dataset/nuclear_fuel_bundle_dataset/`:

- `data/` — 302 images in one folder (the report counts 307 collected)
- `data_sorted/` — images split by shape (144 cylinder, 77 rectangular, 55 octagon)

Steps (in `code/finetune_stable_diffusionv1-4_nuclear_fuel__bundle.ipynb`):

1. **Zip and load the images.** The notebook expects them zipped on Google Drive (`nuclear_fuel_bundle_dataset_2.zip`) and extracts the `data/` subfolder. Change `zip_file_path` if you keep them elsewhere, or point `destination_dir` straight at `dataset/nuclear_fuel_bundle_dataset/data`.
2. **Augment.** Each image gets a random horizontal flip, a brightness jitter (±20%) and a random resized crop to 256×256 (scale 0.8–1.0).
3. **Write `metadata.csv`.** The helper writes a `file_name,text` CSV with a default caption. **Then edit the captions by hand** so each image says its shape:
   - `cylinder nuclear fuel bundle`
   - `rectangular nuclear fuel bundle`
   - `octagon nuclear fuel bundle`

   This shape-specific labelling is the single most important step — training with the generic caption "nuclear fuel bundle" produced mixed, unstructured images.
4. **Push to the Hugging Face Hub** as an `imagefolder` dataset:

   ```python
   from datasets import load_dataset
   ds = load_dataset("imagefolder", data_dir="./augmented_nuclear_fuel_bundle_2", drop_labels=True)
   ds.push_to_hub("<your-hf-username>/augmented_nuclear_fuel_bundle_dataset_2")
   ```

## 3. Fine-tune

```bash
accelerate config default --mixed_precision fp16

export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export DATASET_NAME="<your-hf-username>/augmented_nuclear_fuel_bundle_dataset_2"
export OUTPUT_DIR="nuclear_fuel_bundle-model-v1-4_augmented_3000"

accelerate launch diffusers/examples/text_to_image/train_text_to_image.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --use_ema \
  --resolution=256 --center_crop --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --mixed_precision="fp16" \
  --max_train_steps=3000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --checkpointing_steps=6000 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --use_8bit_adam \
  --output_dir=$OUTPUT_DIR
```

- `--max_train_steps` is the "epochs" number used throughout the project (1000 / 2000 / 3000 / 5000). **3000 gave the best results**; 5000 overfits.
- 8-bit Adam, gradient checkpointing, fp16 and batch size 1 with 4-step accumulation keep memory low enough for a T4.
- Approximate run times on a T4: 2000 steps ≈ 53 min, 3000 ≈ 1 h 20 min, 5000 ≈ 2 h 13 min.

## 4. Generate images

```python
import torch, os
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    "model/nuclear_fuel_bundle-model-v1-4_augmented_3000", torch_dtype=torch.float16
).to("cuda")

os.makedirs("generated/cylinder", exist_ok=True)
for i in range(100):
    pipe("cylinder nuclear fuel bundle").images[0].save(f"generated/cylinder/img_{i+1}.png")
```

Repeat for `rectangular` and `octagon`. The project generated 100 images per shape per model; they are in `output_images/<N> epochs/<shape>_nuclear_fuel_bundle_images/`.

## 5. Evaluate (FID and Inception Score)

Open `code/nuclear_bundle_images_evaluation.ipynb`.

1. Set `real_images_dir` to a folder of real images arranged by class, e.g. `dataset/nuclear_fuel_bundle_dataset/data_sorted/` (the notebook unzips `nuclear_fuel_bundle_dataset_sorted.zip` from Drive into `./realdata/`). Remove any `.ipynb_checkpoints` folder inside it.
2. Set `generated_images_dir` to one model's output folder, e.g. `output_images/3000 epochs/`.
3. Run `calculate_fid_and_is(...)`. Images are resized and centre-cropped to 256×256; FID uses 2048-d Inception features.
4. The last cell plots FID and IS against training steps.

## Model weights

The weight files are **not committed** because each is over GitHub's 100 MB limit:

| File | Size |
|---|---|
| `unet/diffusion_pytorch_model.safetensors` | 3.4 GB |
| `safety_checker/model.safetensors` | 1.2 GB |
| `text_encoder/model.safetensors` | 246 MB |
| `vae/diffusion_pytorch_model.safetensors` | 167 MB |

To use the model, copy these four files into the matching subfolders of `model/nuclear_fuel_bundle-model-v1-4_augmented_3000/` (they are listed in `.gitignore`, so they won't be committed by accident), or retrain with step 3. To share them, upload the whole model folder to the Hugging Face Hub, which handles large files:

```bash
huggingface-cli upload <your-hf-username>/nuclear-fuel-bundle-sd-v1-4 model/nuclear_fuel_bundle-model-v1-4_augmented_3000
```
