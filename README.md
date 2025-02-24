# Interpreting Variational Autoencoders with Fuzzy Logic
*A Step Towards Interpretable Deep Learning-Based Fuzzy Classifiers*

## Overview

This repository contains the MATLAB implementation for the research presented in:

> **Interpreting Variational Autoencoders with Fuzzy Logic: A Step Towards Interpretable Deep Learning-Based Fuzzy Classifiers**  
> *Kutay Bōlat, Tufan Kumbasar*  
> Presented at **WCCI 2020**  

### 🔹 What is this research about?
The study introduces a **Deep Learning-Based Fuzzy Classifier (DL-FC)** that combines:
- **β-Variational Autoencoders (β-VAEs)** for disentangled latent space representation.
- **Fuzzy Sets (FSs) and Fuzzy Logic Systems (FLSs)** for linguistic interpretability.
- **A hybrid classification approach** that merges deep learning with fuzzy logic.

The method is tested on the **MNIST dataset**, demonstrating competitive classification performance while offering **interpretable fuzzy rules**.

---

## 🔬 Summary of the Approach

1. **β-VAE Training:**  
   - A **β-Variational Autoencoder** is trained to learn a disentangled latent space.
   - The latent representations capture semantic information from the dataset.

2. **Latent Space Clustering:**  
   - The latent space is clustered using **Fuzzy C-Means (FCM)** to generate fuzzy sets.

3. **Fuzzy Classifier Design:**  
   - A **Takagi-Sugeno-Kang (TSK) fuzzy system** is built using the generated fuzzy sets.
   - The fuzzy classifier is trained using deep learning methods.

4. **Interpretability Analysis:**  
   - The study explores **latent traversals, heatmaps, and fuzzy set analysis** to provide linguistic interpretations.
   - For the first time, the **latent space of a β-VAE is defined with fuzzy sets**, leading to **interpretable deep learning**.

---

## 📊 Key Findings

- **Classification Performance:**  
  - The DL-FC achieves **competitive accuracy** comparable to deep neural networks.
  - The method uses **only a few fuzzy rules** while maintaining high performance.

- **Interpretability:**  
  - Specific latent dimensions correspond to **linguistic variables** (e.g., "Inclination angle of the digit", "Circularity of the digit").
  - **Heatmaps and latent traversals** reveal meaningful **semantic representations**.

These findings highlight the potential of fuzzy logic in designing **interpretable deep learning models**.

---

## 📄 Citation
If you use or refer to this work, please cite:

```bibtex
@inproceedings{bolat2020interpreting,
  title={Interpreting Variational Autoencoders with Fuzzy Logic: A step towards interpretable deep learning based fuzzy classifiers},
  author={B{\"o}lat, Kutay and Kumbasar, Tufan},
  booktitle={2020 IEEE International Conference on Fuzzy Systems (FUZZ-IEEE)},
  pages={1--7},
  year={2020},
  organization={IEEE}
}
