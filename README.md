# Multi-Modal Price Prediction System

**Amazon ML Challenge**

An end-to-end multimodal deep learning pipeline designed to predict e-commerce product prices from noisy, unstructured catalog text, visual representations, and extracted metadata.

---

## 📌 Project Overview

E-commerce product pricing presents distinct machine learning challenges:

* **Multimodal Data:** Catalog items combine unstructured descriptions, images, and embedded tabular attributes.


* **Heavy-Tailed Skewness:** Product prices in the dataset span from **$0.13 to $2,796.00**, with ~85% of listings priced under $40 and a long tail of high-value products.


* **Metric Sensitivity:** Evaluation uses **SMAPE** (Symmetric Mean Absolute Percentage Error) and **MAE**, requiring balanced penalties across relative percentage errors without allowing cheap products to dominate optimization.



This project implements an offline multimodal extraction and caching pipeline, a **1,046-dimensional feature space**, deep MLP ensembles trained on log-transformed targets with custom loss functions, and a dynamic Mixture-of-Experts (MoE) price-tier routing architecture.

---

## 🏗️ Architecture & Pipeline Flow

```
[ Raw Catalog Data (~45 GB) ]
       │
       ├──► Cloud VM Processing: OpenAI CLIP (ViT-B/32) Image & Text Encodings
       │         ├── 512-dim Precomputed Image Embeddings
       │         └── 512-dim Normalized Text Embeddings
       │
       ├──► Cross-Modal Similarity: Cosine Alignment (3 dims)
       │
       └──► Regex Parsing: Unit Normalization & Text Statistics (19 dims)
                 │
                 ▼
       [ Unified 1,046-dim Feature Vector ]
                 │
       ┌─────────┴─────────────────────────────────────────┐
       ▼                                                   ▼
[ Baseline 10-Model MLP Ensemble ]           [ Dynamic MoE Routing Pipeline ]
• Log-Transformed Price Targets (log1p)       • Coarse Router (87.4% accuracy)
• Huber Loss / SmoothL1 Optimization         • Tier 1: Affordable (<$40) [10 MLPs]
• Mixed Precision (AMP) Training              • Tier 2: Mid-Range ($40-$85) [7 MLPs]
• Reverted via expm1                          • Tier 3: Premium (>$85) [Top 3 MLPs]
                                             • Weighted SMAPE Optimization

```

---

## 🔬 Feature Engineering Breakdown (1,046 Dimensions)

The unified feature vector concatenates dense representations with structured business signals:

1. **Vision Embeddings (512 dims):** Normalized visual embeddings capturing product appearance.


2. **Text Embeddings (512 dims):** Catalog text truncated to 300 characters and encoded via OpenAI's frozen `CLIP (ViT-B/32)`.


3. **Cross-Modal Alignment (3 dims):** Row-wise cosine similarity between normalized image and text embeddings ($S = v_{\text{img}} \cdot v_{\text{txt}}$) alongside threshold indicators ($S > 0.8$ and $S < 0.5$).


4. **Structured Tabular Features (19 dims):**
   * **Regex Unit Normalization:** Parses explicit (`Value:`, `Unit:`) and implicit patterns, mapping irregular strings (`fl oz`, `ounces`, `grams(gm)`, `per box`, `carton`) into base units (`qty_base` in mL, g, or item count).
   * **Text Statistics:** Word count, character count, average word length, bullet point count, and digit counts.
   * **Keyword Multipliers:** Boolean flags indicating premium attributes (`is_organic`, `is_gluten_free`, `is_sugar_free`, `is_vegan`, `is_new`, `has_pack_word`, `has_bundle`).

$$\text{Total Dimension} = 512 + 512 + 3 + 19 = 1,046\text{ dimensions}$$

---

## 📊 Experimental Results & Evaluation

All evaluations were executed on a strict held-out validation set (80/20 split: 60,000 train / 15,000 validation).

| Model / Strategy | Objective / Loss Function | Validation MAE | Validation SMAPE | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **10-Model Baseline Ensemble** | SmoothL1 Loss on $\log(1 + x)$ | **$12.14** | **53.37%** | **Selected for Test Inference** |
| **Specialist Ensembles with Routing** | Weighted SMAPE (5× on Premium) | $13.05 | 54.65% | **87.4% routing accuracy** |
| **Level-2 Random Forest Stacking** | MAE on 18 Meta-Features | $12.59 | — | Level-2 meta-learner overfit |
| **Level-2 Lasso Regression** | L1 Regularized Meta-Model | $13.08 | — | Collinear base models |
| **Level-2 Ridge Regression** | L2 Regularized Meta-Model | $13.11 | — | Collinear base models |

---

## 🧠 Key Engineering Takeaways & Post-Mortem

### 1. Offline Embedding Serialization over End-to-End Fine-Tuning

Running iterative end-to-end forward and backward passes through a full Vision-Language model across a 45 GB product archive created severe compute and memory bottlenecks. Decoupling feature extraction by caching static 512-dim CLIP vectors shifted the operational constraint to RAM and I/O, accelerating downstream tabular experiments, custom loss iterations, and ensembling.

### 2. Why Uniform Averaging Beat Meta-Learning Stacking

Level-2 meta-regressors (Ridge, Lasso, Random Forest) trained on 18 meta-features underperformed simple averaging ($12.59 vs $12.28 MAE). Because the 10 base MLPs shared identical architectures and inputs, their outputs were highly collinear. The meta-learners fit validation residuals rather than finding complementary predictive signals.

### 3. Mixture-of-Experts Diagnosis & Fallback

The dynamic price router achieved **87.4% routing classification accuracy** across three price tiers. However, overall validation SMAPE degraded slightly compared to the baseline (54.65% vs. 53.37%).

* **Failure Analysis:** Premium items ($> \$85$) made up only ~5.1% of the data. The specialized premium MLP ensemble regressed toward the mean, predicting an average of **$48.95** on items actually averaging **$142.96**.


* **Loss Geometry:** Under SMAPE, extreme under-predictions on high-value products incur severe percentage penalties.


* **Decision:** The pipeline automatically fell back to the generalized baseline ensemble for test set inference, avoiding distribution boundary errors.



---

## 📂 Repository Structure

```
├── 01_data_exploration_and_preprocessing.ipynb      # EDA, image downloading, Regex normalization & text baselines
├── 02_clip_embeddings_clustering_and_stacking.ipynb # CLIP embeddings, 1,046-dim fusion, baseline MLPs & stacking
├── 03_test_dataset_pipeline.ipynb                   # Test feature extraction pipeline & baseline ensemble inference
├── 04_mixture_of_experts_routing.ipynb              # Price-tier specialists, dynamic router & MoE evaluation
├── requirements.txt                                 # Environment dependencies
└── README.md                                        # Project documentation
```

### 📓 Notebook Breakdown

* **`01_data_exploration_and_preprocessing.ipynb` — Data Exploration, Cleaning & Early Baselines**
  * **Exploratory Data Analysis:** Profiles price distribution skewness ($0.13 to $2,796) and performs stratified quantile sampling to retain long-tail high-value representation.
  * **Image Downloader:** Multi-threaded downloader with retry loops, timeout handling, and resume checks to fetch catalog images from URLs.
  * **Regex Feature Engineering:** Parses unstructured catalog text into structured attributes (`Item Name`, `Bullet Points`, `Product Description`) and canonicalizes irregular quantity strings (`fl oz`, `grams`, `pack of X`, `carton`) into base units (`qty_base` in mL, g, or item count).
  * **Early Baselines:** Evaluates initial text regression baselines using `DistilBERT` (Hugging Face Trainer) and `SentenceTransformer (all-MiniLM-L6-v2)` concatenated with 19 structured features.

* **`02_clip_embeddings_clustering_and_stacking.ipynb` — Multimodal Fusion, Baseline Ensemble & Stacking**
  * **1,046-Dim Feature Construction:** Combines 512-dim normalized OpenAI `CLIP (ViT-B/32)` text embeddings, 512-dim visual embeddings, 3-dim cross-modal cosine alignment features ($S = v_{\text{img}} \cdot v_{\text{txt}}$ plus confidence indicators), and 19 structured tabular attributes.
  * **10-Model Baseline MLP Ensemble:** Trains 10 independently seeded 4-layer regression MLPs with BatchNorm and Dropout on $\log(1 + x)$ prices using SmoothL1 loss, achieving the primary winning validation result (**$12.14 MAE**, **53.37% SMAPE**).
  * **Level-2 Stacking Experiments:** Extracts 18 meta-features from base models and evaluates Ridge, Lasso, and Random Forest meta-learners, proving that collinearity causes meta-regressors to underperform uniform averaging.
  * **Unsupervised Clustering:** Uses K-Means clustering on CLIP text embeddings to explore latent catalog product groupings.

* **`03_test_dataset_pipeline.ipynb` — Test Processing & Final Submission Generation**
  * **Deterministic Pipeline Application:** Applies the identical Regex normalization, attribute parsing, and missing-value imputation pipeline to the 75,000 unlabelled test catalog items.
  * **Test Feature Matrix:** Computes 512-dim CLIP text embeddings for the test set, merges precomputed test visual embeddings, evaluates cross-modal cosine alignment, and constructs the 1,046-dim test tensor with strict sample ID alignment.
  * **Ensemble Test Inference:** Generates predictions via the 10-model baseline MLP ensemble with log-space bounds clipping and inverse exponential scaling (`expm1`), producing the final `submission_predictions.csv` and analyzing cross-model prediction variance.

* **`04_mixture_of_experts_routing.ipynb` — Dynamic Mixture-of-Experts (MoE) & Specialist Routing**
  * **Price-Tier Segmentation:** Partitions products into 3 economic tiers: Affordable (<$40, ~85%), Mid-Range ($40–$85, ~10%), and Premium (>$85, ~5%).
  * **Custom Loss Functions:** Trains specialized MLP ensembles per price bucket utilizing a custom **Weighted SMAPE loss** ($5\times$ penalty for premium items, $1.5\times$ for mid-range) and regularized architectures for data-sparse premium items.
  * **Routing Evaluation & Post-Mortem:** Benchmarks a dynamic two-stage routing pipeline (87.4% coarse routing accuracy) against the baseline ensemble, identifying regression-to-the-mean issues in the premium bucket and automating defensive fallback to the baseline ensemble.

---

Core dependencies:

* `torch >= 2.0.0`
* `torchvision`
* `clip @ git+https://github.com/openai/CLIP.git`
* `pandas`
* `numpy`
* `scikit-learn`
* `tqdm`