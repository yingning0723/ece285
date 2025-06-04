# DDPM for Fine-Grained Fashion Generation

This is the official repository for our ECE 285 project implementing a **Conditional Denoising Diffusion Probabilistic Model (DDPM)** on the Fashion-MNIST dataset for fine-grained image generation.

---

## 📂 Project Structure

- `train.py` – Train the conditional DDPM model.
- `sample.py` – Generate samples from the trained model and visualize them.
- `models/` – Contains the conditional UNet architecture with FiLM layers.
- `utils/` – Dataset loaders and helper functions.
- `checkpoints/` *(branch)* – Stores all trained model checkpoints.
- `generated-images/` *(branch)* – Contains generated sample results used in the final report.

---

## 🧠 Model Overview

We implement a **Conditional UNet-based DDPM** with:
- 3-level downsampling and upsampling architecture.
- FiLM (Feature-wise Linear Modulation) layers for label conditioning.
- Sinusoidal timestep embeddings.
- Configurable noise schedule (linear or cosine).
- Loss: Mean Squared Error (MSE) between predicted and true noise.

---

## 🛠️ Training

To train the model from scratch:

```bash
python train.py --dataset-name fashion-mnist --batch-size 64 --epochs 40 --lr 1e-4

