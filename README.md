# UniPrompt-CL: Unified Prompt Pool for Continual Learning

Official PyTorch implementation of **UniPrompt-CL**, a framework with a **Unified Prompt Pool** designed for continual learning (CL) in the medical domain.  
This work has been submitted to **ECCV 2026**.

---

## Table of Contents
- [Overview](#-overview)
- [Key Contributions](#-key-contributions)
- [Architecture Overview](#-architecture-overview)
- [Setup](#-setup)
- [Data Preparation](#-data-preparation)
- [Training](#-training)
- [Results](#-results)
- [Reproducibility](#-reproducibility)
- [Citation](#-citation)
- [License](#-license)

---

## 📖 Overview

While state-of-the-art models achieve strong performance on large-scale natural image datasets, medical AI faces distinct challenges:

- 📌 Privacy and ethical constraints → centralized training is often infeasible
- 📌 Domain shift → caused by device heterogeneity and demographic/hospital differences
- 📌 Catastrophic forgetting → knowledge erasure in sequential learning

To address these issues, UniPrompt-CL introduces:
- 🔹 Unified Prompt Pool: integrates prompts across layers into a single pool, reducing redundancy
- 🔹 Minimal Prompt Expansion: adds only a small set of prompts to adapt to new domains
- 🔹 Stabilizing regularization: ensures stable training and prevents prompt overlap
- 🔹 DINOv2-base backbone (~86.6M parameters): balances expressiveness and efficiency

## 💡 Why General PCL Fails and Why We Need Medical-domain Prompt-based CL

- **Data constraints & privacy issues**  
  Medical data cannot be freely shared due to strict ethical and legal restrictions. Hospitals must train models locally, and rehearsal-based methods that store or reuse past data are infeasible under such privacy-sensitive conditions.

- **Specific characteristics of medical imaging**  
  Unlike natural images, medical images are collected under standardized acquisition protocols, which means that diverse viewpoints and compositions rarely occur. Instead, distribution shifts mainly arise from patient-specific physiological variations (e.g., subtle color/texture changes) and inter-hospital or inter-device discrepancies. Therefore, rather than broadly covering general features as in natural image prompts, domain-tailored prompts capable of capturing fine-grained patterns such as vessel and nerve structures are required.

- **Limitations of existing CL approaches**  
  Regularization, architecture, and rehearsal-based methods are not well-suited to the medical domain due to complexity, resource demands, and privacy concerns. Furthermore, existing PCL methods originally designed for natural images often degrade in performance on medical datasets and incur unnecessary computational costs (e.g., multiple ViT inferences).

- **Necessity of medical-domain PCL**  
By combining a fixed backbone (e.g., ViT, DINOv2) with lightweight prompt learning, models can preserve prior knowledge while adapting efficiently to new domains. Such domain-specific prompts effectively mitigate catastrophic forgetting, capture fine-grained clinical variations, and provide a privacy-friendly and practical CL strategy for real-world medical AI.


## 🚀 Key Contributions
- We highlight the need for a medical-domain-specific PCL method and, accordingly, propose the following contributions.
1. Unified Prompt Pool  
   - Consolidates per-layer prompts into a unified pool  
   - Enhances fine-grained and stable feature representation

2. Few Prompt Expansion  
   - Expands the pool with only 20% new prompts per stage  
   - Maintains efficiency while enabling new knowledge acquisition

3. Lightweight yet strong performance  
   - Improves accuracy by +10% and F1 by +9 points over baselines  
   - Achieves results with a single ViT inference (vs. dual inference in prior work)  
   - Reduces FLOPs by approximately 30%


---

## 🧩 Architecture Overview

The following figures summarize the core ideas of UniPrompt-CL: unified prompt pool, minimal expansion, and single-inference pipeline.

<figure>
  <img src="image/figure1.png" alt="Unified prompt pool and minimal expansion overview" />
  <figcaption><b>Figure 1.</b> Overview of the Unified Prompt Pool: Unlike other methods, all layers share a single unified prompt pool, enabling efficient knowledge exchange. As the training stages progress, only a small number of new prompts are added, while the previously learned prompts remain fixed.</figcaption>
</figure>

<figure>
  <img src="image/figure2.png" alt="Prompt distribution comparison between OS-Prompt and UniPrompt-CL" />
  <figcaption><b>Figure 2.</b> Visualization of UniPrompt-CL’s prompt learning strategy. (a) Prompt distributions: OS-Prompt (independent pools) produces widely scattered prompts suited to broad natural-image diversity, whereas our integrated pool (brown) forms compact, domain-tailored clusters that capture subtle, medically relevant variations across patients and devices. (b) Independence of newly added prompts: stage-wise similarity matrices exhibit a clear diagonal, indicating that new prompts learn non-redundant information, efficiently absorbing novel knowledge while avoiding overlap with previously learned prompts. Overall, Figure 2 evidences fine-grained, medical-domain–oriented prompt learning that improves continual learning without exacerbating catastrophic forgetting.</figcaption>
</figure>

<figure>
  <img src="image/figure3.png" alt="Representative examples from natural-image PCL and medical datasets" />
  <figcaption><b>Figure 3.</b> This figure presents representative examples of conventional natural images and the medical images used in this study. As illustrated, natural images exhibit diverse viewpoints and shapes, whereas medical data generally maintain consistent angles and structures. Moreover, the identification of pathological conditions requires careful recognition of subtle color differences and small lesions. These examples support the claims made in this work.</figcaption>
</figure>

---

## 🛠 Setup

- Python 3.8
- PyTorch 2.4.1 + CUDA 11.8
- NVIDIA V100 × 2 (≈8GB VRAM per experiment)

Recommended installation:

```bash
pip3 install -r requirements.txt  
```

---

## 🗂 Data Preparation

Datasets:
- [APTOS 2019 Blindness Detection](https://kaggle.com/competitions/aptos2019-blindness-detection)
- [DDR](https://github.com/nkicsl/DDR-dataset)
- [DRD](https://kaggle.com/competitions/diabetic-retinopathy-detection)

Files requiring path configuration:
- `./dataloader/atop_dataloader.py`
- `./dataloader/DDR_dataloader.py`
- `./run_train.sh`

Fill in local dataset paths where indicated in the code comments.  
CSV samples are provided under `dataloader/`.

---

## 🎯 Training

1) Configure dataset paths  
2) Start training via script:

```bash
bash run_train.sh
```

Key directories:
- `train/train.py`, `train/ddp_train.py`: training loop
- `network/model.py`, `network/loss.py`: model and loss functions
- `util/early_stopping.py`, `util/scheduler.py`, `util/utils.py`: utilities

---

## 📊 Results
Explanations for each table are provided in detail in the paper!
If any part is unclear, please refer to the paper for further descriptions.
Thank you.

### 🔹 Table 1. Final Accuracy (Acc) and F1-Score (F1) Results After the Final Stage and Performance Comparison with Other Prompt-based CL Models. The symbol † denotes our proposed model. (bold indicates the highest performance; scores are mean values with negligible deviations.)

| Ref. | Model | APTOS (Acc/F1) | DDR (Acc/F1) | DRD (Acc/F1) |
|------|-------|----------------|--------------|--------------|
| ECCV 2024 | OS [16] | 0.687 / 0.637 | 0.693 / 0.648 | 0.619 / 0.568 |
| ECCV 2024 | OS++ [16] | 0.743 / 0.686 | 0.697 / 0.655 | 0.623 / 0.565 |
| CVPR 2023 | Coda-Prompt [30] | 0.682 / 0.646 | 0.721 / 0.697 | 0.663 / 0.557 |
| CVPR 2022 | L2P [35] | 0.353 / 0.174 | 0.421 / 0.194 | 0.603 / 0.252 |
| ECCV 2022 | Dual-Prompt [34] | 0.363 / 0.185 | 0.435 / 0.222 | 0.604 / 0.259 |
| **ECCV 2026** | **UniPrompt-CL (Ours)** | **0.849 / 0.761** | **0.772 / 0.723** | **0.701 / 0.656** |

---

### 🔹 Table 2. Stage-wise Performance (Catastrophic Forgetting and FLOPs) : racking and comparing various catastrophic forgetting outcomes during stage progression (where Red indicates data learned at the current step, Blue indicates previously learned data (seen), and Black indicates unseen data) [Accuracy (Acc), F1-score (F1), OS prompt++ (OS++); Horizontal: Training Data, Vertical: Evaluation Data]. Additionally, the FLOPs row indicates the amount of computing resources used. The symbol † denotes our proposed model.

<table>
  <thead>
    <tr>
      <th rowspan="2">Training</th>
      <th rowspan="2">Dataset</th>
      <th colspan="6">OS-Prompt++ (Dual inference)</th>
      <th colspan="6">UniPrompt-CL (Single inference)</th>
    </tr>
    <tr>
      <th colspan="2">APTOS</th>
      <th colspan="2">DDR</th>
      <th colspan="2">DRD</th>
      <th colspan="2">APTOS</th>
      <th colspan="2">DDR</th>
      <th colspan="2">DRD</th>
    </tr>
    <tr>
      <th></th>
      <th></th>
      <th>Acc</th><th>F1</th>
      <th>Acc</th><th>F1</th>
      <th>Acc</th><th>F1</th>
      <th>Acc</th><th>F1</th>
      <th>Acc</th><th>F1</th>
      <th>Acc</th><th>F1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Stage 1</td><td>APTOS</td>
      <td><span style="color:#d62728;">0.868</span></td><td><span style="color:#d62728;">0.753</span></td>
      <td>0.565</td><td>0.474</td>
      <td>0.409</td><td>0.354</td>
      <td><span style="color:#d62728;">0.901</span></td><td><span style="color:#d62728;">0.767</span></td>
      <td>0.601</td><td>0.447</td>
      <td>0.453</td><td>0.381</td>
    </tr>
    <tr>
      <td>Stage 2</td><td>DDR</td>
      <td><span style="color:#1f77b4;">0.707</span></td><td><span style="color:#1f77b4;">0.638</span></td>
      <td><span style="color:#d62728;">0.797</span></td><td><span style="color:#d62728;">0.748</span></td>
      <td>0.508</td><td>0.413</td>
      <td><span style="color:#1f77b4;">0.866</span></td><td><span style="color:#1f77b4;">0.663</span></td>
      <td><span style="color:#d62728;">0.878</span></td><td><span style="color:#d62728;">0.844</span></td>
      <td>0.636</td><td>0.534</td>
    </tr>
    <tr>
      <td>Stage 3</td><td>DRD</td>
      <td><span style="color:#1f77b4;">0.743</span></td><td><span style="color:#1f77b4;">0.686</span></td>
      <td><span style="color:#1f77b4;">0.697</span></td><td><span style="color:#1f77b4;">0.655</span></td>
      <td><span style="color:#d62728;">0.623</span></td><td><span style="color:#d62728;">0.565</span></td>
      <td><span style="color:#1f77b4;">0.849</span></td><td><span style="color:#1f77b4;">0.761</span></td>
      <td><span style="color:#1f77b4;">0.772</span></td><td><span style="color:#1f77b4;">0.723</span></td>
      <td><span style="color:#d62728;">0.701</span></td><td><span style="color:#d62728;">0.656</span></td>
    </tr>
  </tbody>
  <tfoot>
    <tr>
      <th colspan="2">FLOPs</th>
      <td colspan="6">66.42 GFLOPs</td>
      <td colspan="6">44.17 GFLOPs</td>
    </tr>
  </tfoot>
</table>

---

### 🔹 Table 3. Ablation – Effect of Backbone (DINOv2) : We compare the results of fatal forgetting during the stepwise progression of the baseline model, the introduction of a stronger backbone, and the methodology of this study. Through these exclusion studies, we highlight the importance of a good backbone in PCL and show that there is room for further improvement. It can be interpreted in the same way as Table 2.

<table>
  <caption><b>OS-Prompt++ (Original)</b></caption>
  <thead>
    <tr>
      <th rowspan="2"></th>
      <th colspan="2">APTOS</th>
      <th colspan="2">DDR</th>
      <th colspan="2">DRD</th>
    </tr>
    <tr>
      <th>Acc</th><th>F1</th>
      <th>Acc</th><th>F1</th>
      <th>Acc</th><th>F1</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>APTOS</td><td><span style="color:#d62728;">0.868</span></td><td><span style="color:#d62728;">0.753</span></td><td>0.565</td><td>0.474</td><td>0.409</td><td>0.354</td></tr>
    <tr><td>DDR</td><td><span style="color:#1f77b4;">0.707</span></td><td><span style="color:#1f77b4;">0.638</span></td><td><span style="color:#d62728;">0.797</span></td><td><span style="color:#d62728;">0.748</span></td><td>0.508</td><td>0.413</td></tr>
    <tr><td>DRD</td><td><span style="color:#1f77b4;">0.743</span></td><td><span style="color:#1f77b4;">0.686</span></td><td><span style="color:#1f77b4;">0.697</span></td><td><span style="color:#1f77b4;">0.655</span></td><td><span style="color:#d62728;">0.623</span></td><td><span style="color:#d62728;">0.565</span></td></tr>
  </tbody>
</table>

<table>
  <caption><b>OS-Prompt++ (Add DINOv2)</b></caption>
  <thead>
    <tr>
      <th rowspan="2"></th>
      <th colspan="2">APTOS</th>
      <th colspan="2">DDR</th>
      <th colspan="2">DRD</th>
    </tr>
    <tr>
      <th>Acc</th><th>F1</th>
      <th>Acc</th><th>F1</th>
      <th>Acc</th><th>F1</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>APTOS</td><td><span style="color:#d62728;">0.918</span></td><td><span style="color:#d62728;">0.823</span></td><td>0.608</td><td>0.520</td><td>0.492</td><td>0.467</td></tr>
    <tr><td>DDR</td><td><span style="color:#1f77b4;">0.732</span></td><td><span style="color:#1f77b4;">0.604</span></td><td><span style="color:#d62728;">0.849</span></td><td><span style="color:#d62728;">0.828</span></td><td>0.625</td><td>0.563</td></tr>
    <tr><td>DRD</td><td><span style="color:#1f77b4;">0.754</span></td><td><span style="color:#1f77b4;">0.690</span></td><td><span style="color:#1f77b4;">0.763</span></td><td><span style="color:#1f77b4;">0.721</span></td><td><span style="color:#d62728;">0.668</span></td><td><span style="color:#d62728;">0.585</span></td></tr>
  </tbody>
</table>

<table>
  <caption><b>UniPrompt-CL (Ours)</b></caption>
  <thead>
    <tr>
      <th rowspan="2"></th>
      <th colspan="2">APTOS</th>
      <th colspan="2">DDR</th>
      <th colspan="2">DRD</th>
    </tr>
    <tr>
      <th>Acc</th><th>F1</th>
      <th>Acc</th><th>F1</th>
      <th>Acc</th><th>F1</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>APTOS</td><td><span style="color:#d62728;">0.901</span></td><td><span style="color:#d62728;">0.767</span></td><td>0.601</td><td>0.447</td><td>0.453</td><td>0.381</td></tr>
    <tr><td>DDR</td><td><span style="color:#1f77b4;">0.866</span></td><td><span style="color:#1f77b4;">0.663</span></td><td><span style="color:#d62728;">0.878</span></td><td><span style="color:#d62728;">0.844</span></td><td>0.636</td><td>0.534</td></tr>
    <tr><td>DRD</td><td><span style="color:#1f77b4;">0.849</span></td><td><span style="color:#1f77b4;">0.761</span></td><td><span style="color:#1f77b4;">0.772</span></td><td><span style="color:#1f77b4;">0.723</span></td><td><span style="color:#d62728;">0.701</span></td><td><span style="color:#d62728;">0.656</span></td></tr>
  </tbody>
</table>

---

### 🔹 Table 4. Ablation – Continual Learning Metrics (AvgACC, BWT, AvgF)

| Method | AvgACC ↑ | BWT ↑ | AvgF ↓ |
|--------|----------|-------|--------|
| OS-Prompt++ (Original) | 0.769 | −0.113 | 0.113 |
| OS-Prompt++ (+DINOv2) | 0.812 | −0.125 | 0.125 |
| **UniPrompt-CL (Ours)** | **0.849** | **−0.079** | **0.079** |

---

### 🔹 Table 5. Ablation – Prompt Expansion Ratio

| #Prompt Extensions | Stage | APTOS (Acc/F1) | DDR (Acc/F1) | DRD (Acc/F1) |
|--------------------|-------|----------------|--------------|--------------|
| **50 (10%)** | Stage 1 | 0.898 / 0.765 | 0.592 / 0.440 | 0.446 / 0.372 |
|                  | Stage 2 | 0.830 / 0.659 | 0.868 / 0.852 | 0.630 / 0.534 |
|                  | Stage 3 | 0.803 / 0.751 | 0.739 / 0.706 | 0.692 / 0.659 |
| **100 (20%) ♣** | Stage 1 | 0.901 / 0.767 | 0.601 / 0.447 | 0.453 / 0.381 |
|                  | Stage 2 | 0.866 / 0.663 | 0.878 / 0.844 | 0.636 / 0.534 |
|                  | Stage 3 | 0.849 / 0.761 | 0.772 / 0.723 | 0.701 / 0.656 |
| **150 (30%)** | Stage 1 | 0.901 / 0.774 | 0.608 / 0.469 | 0.470 / 0.408 |
|                  | Stage 2 | 0.816 / 0.668 | 0.869 / 0.856 | 0.637 / 0.543 |
|                  | Stage 3 | 0.836 / 0.757 | 0.757 / 0.709 | 0.694 / 0.636 |

---

### 🔹 Table 6. Ablation – Effect of Regularization (Ls) and λ

| Ls | λ | FAA | FAF |
|----|---|-----|-----|
| - | - | 0.754 | 0.701 |
| ✓ | 0.01 | 0.777 | 0.705 |
| ✓ | 0.001 | 0.775 | 0.713 |
| ✓ | 0.0001 | 0.765 | 0.723 |

---

## ⚙️ Reproducibility

**Environment**:  
- Python 3.8  
- PyTorch 2.4.1 + CUDA 11.8  
- NVIDIA V100 × 2 (≈8GB VRAM per experiment)

**Training Setup**:  
- Epochs: 100  
- Batch size: 64  
- Input size: 224 × 224  
- Optimizer: AdamW  
- Scheduler: Cosine LR  
- Number of prompts: 500 (dim 768)

---


---

## 🧾 License

This repository is temporarily open for review purposes. We encourage use for research and non-commercial purposes.

