# 🚀 From Scratch: PCA & Naive Bayes Comparative Analysis

A rigorous exploration into **Dimensionality Reduction** and **Probabilistic Classification**, featuring custom implementations of PCA and Naive Bayes to evaluate performance across Numerical and Categorical domains.

---


## 📌 Project Overview
This project benchmarks the performance of **Gaussian** and **Categorical Naive Bayes** using two iconic datasets: **UCI Breast Cancer** (Numerical) and **UCI Mushroom** (Categorical). The core objective was to analyze the trade-offs between **Feature Selection**, **PCA (Feature Reduction)**, and **Baseline** models.

<img width="1389" height="589" alt="comparison" src="https://github.com/user-attachments/assets/b7d00d0f-ed9e-475b-8126-2f9d30ca2cb9" />

## 🛠️ Key Features
* **Pure NumPy Implementation:** PCA built from the ground up using Eigen-decomposition ($\Sigma v = \lambda v$).
* **Custom Naive Bayes Suite:** Manual implementation of Gaussian (Continuous) and Categorical (Discrete) Naive Bayes with Laplace Smoothing.
* **Experimental Benchmarking:**
    * **Experiment 0:** Baseline (Full Feature Space).
    * **Experiment A:** Feature Selection (SelectKBest).
    * **Experiment B:** PCA Dimensionality Reduction.

## 📂 Datasets & Justification
1.  **Breast Cancer Wisconsin (Numerical):** 30 numeric features. Ideal for **Gaussian NB** as standardized features often approximate a normal distribution.
2.  **Mushroom Dataset (Categorical):** 22 discrete features. Ideal for **Categorical NB** to estimate class-conditional probabilities based on category counts.

---

## 📈 Results & Performance Matrix

| Experiment | **Breast Cancer** (Gaussian) | **Mushroom** (Categorical) |
| :--- | :---: | :---: |
| **Baseline (All Features)** | **96.49%** | **95.08%** |
| **Feature Selection** | 92.10% | 88.00% |
| **PCA (From Scratch)** | 94.73% | 87.75%* |


---

## 📊 Methodology & Mathematical Insights

### 1. PCA From Scratch
The PCA implementation follows a strict mathematical pipeline:
1.  **Mean Centering:** $X_{centered} = X - \mu$.
2.  **Covariance Matrix:** $\Sigma = \frac{1}{n} X^T X$.
3.  **Eigen-Decomposition:** Solving for eigenvalues ($\lambda$) and eigenvectors ($v$).
4.  **Projection:** Transforming data into a lower-dimensional subspace while retaining maximum variance.



### 2. Critical Analysis
* **The Baseline Supremacy:** For both datasets, the Baseline achieved the highest accuracy. This highlights that Naive Bayes thrives on aggregating independent signals; compressing these signals (via PCA) can sometimes blur the distinct evidence needed for classification.
* **Numerical vs. Categorical PCA:** PCA maintained high performance on numerical data (94.73%) because it preserved global variance. However, on categorical data, the forced discretization (rounding) led to a drop, confirming that PCA is mathematically mismatched for non-continuous data.
* **Interpretability vs. Information:** Feature Selection kept real-world traits (like "odor"), making the model explainable, while PCA offered better information retention at the cost of interpretability.

---

## 💻 Technical Stack
* **Language:** Python 3.x
* **Core:** NumPy, Pandas
* **Visualization:** Matplotlib, Seaborn
* **Source Data:** Scikit-learn, UCI Repository

