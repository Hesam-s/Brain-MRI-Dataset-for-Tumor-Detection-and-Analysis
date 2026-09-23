Here is a comprehensive and structured `README.md` summary file designed for your GitHub repository. It integrates all the code modules, datasets, weights, and notebook workflows into a professional documentation template.

---

# Brain Tumor Detection and Classification in MRI Using Hybrid ViT and GRU Model with Explainable AI

---

## 📖 Project Overview

This repository implements an advanced deep learning framework designed to enhance diagnostic accuracy for brain tumors. Due to regional data constraints (modeled after the clinical landscape of Southern Bangladesh and proxy benchmark references like **2023-BrTMHD**), this implementation utilizes the **Brain MRI Dataset for Tumor Detection and Analysis** as a high-fidelity benchmark.

The core of this repository is a **Hybrid Vision Transformer (ViT) and Gated Recurrent Unit (GRU)** architecture, complemented by a robust data preprocessing pipeline and **Explainable AI (XAI)** interpretability tools (LIME & SHAP).

---

## 👥 Research & Development Team

* **Hesam Servati** – `404198788`
* **Mohammad mahdi Nosrati** – `404188074`
* **Amirali Mohammadi** – `404224755`

**Under the Supervision of:**

**Professor Maliheh Sabeti (دکتر ملیحه ثابتی)**

*North Tehran Branch | Azad University*

---

## 📂 Repository Structure

```text
├── dataset/                    # Brain MRI Dataset for Tumor Detection and Analysis
├── model.py                    # PyTorch implementation of the Hybrid ViT-GRU model
├── preprocessing.py            # OpenCV & TensorFlow-based image enhancement pipeline
├── train.eval.ipynb            # Main Jupyter Notebook for training, evaluation, and XAI
├── best_model.pt               # Saved best model state dictionary / checkpoint
└── holdout_weights.weights     # Pre-trained holdout model weights

```

---

## ⚙️ Core Architecture & Modules

### 1. Preprocessing Pipeline (`preprocessing.py`)

To maintain medical imaging fidelity and maximize data utility, the preprocessing module executes:

* **Gaussian Noise Removal:** Applies a $3 \times 3$ Gaussian blur kernel (`apply_gaussian_denoising`) to reduce high-frequency grain while keeping tumor borders sharp.
* **High-Quality Resizing:** Utilizes Lanczos interpolation (`cv2.INTER_LANCZOS4`) to scale spatial dimensions to $224 \times 224$ pixels.
* **Normalization & RGB Mapping:** Scales raw intensity matrices to a clean $[0, 1]$ floating-point range and handles channel replication for ViT compatibility.
* **Chunk-Based Data Augmentation:** Implements an in-place `ImageDataGenerator` pipeline processed in safe memory chunks to prevent system RAM underflows.

### 2. Hybrid Deep Learning Model (`model.py`)

The architecture combines spatial self-attention with sequential context modeling:

* **Patch Embedding:** Uses a 2D Convolutional layer to break the $224 \times 224$ input into localized patches.
* **Vision Transformer (ViT):** Multi-head self-attention blocks (8 encoder layers, 4 heads) to capture global spatial dependencies across MRI slices.
* **Gated Recurrent Unit (GRU):** Models sequential/volumetric depth dependencies over feature sequences.
* **Classification Head:** Fully connected layers optimized for binary/categorical cross-entropy classification (`Tumor` vs. `No Tumor`).

---

## 🚀 Getting Started & Installation

1. **Clone the Repository:**
```bash
git clone https://github.com/your-username/brain-mri-vit-gru.git
cd brain-mri-vit-gru

```


2. **Install Dependencies:**
```bash
pip install torch torchvision torchinfo tensorflow opencv-python scikit-learn pandas matplotlib seaborn lime shap kagglehub

```


3. **Run the Pipeline:**
Open `train.eval.ipynb` in Google Colab or a local Jupyter environment to execute data downloading via `kagglehub`, initiate preprocessing via `preprocessing.py`, initialize the model via `model.py`, and run training/evaluation loops alongside SHAP and LIME interpretations.

---

## 📈 Evaluation & Model Weights

* **Trained Weights:** The repository includes pre-trained model states (`best_model.pt` and `holdout_weights.weights`) for rapid evaluation and inference without requiring full re-training.
* **Explainability (XAI):** Integrated LIME and SHAP visualizations highlight influential pixels and confidence attributions for clinical interpretability.
