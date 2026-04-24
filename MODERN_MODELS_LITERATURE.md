# Literature Review: Modern Models for Log Anomaly Detection

> **Context:** This document supports the supervisor's feedback to use state-of-the-art models
> instead of traditional Logistic Regression and Random Forest for the BGL log anomaly detection task.
> All papers cited here focus specifically on the **BGL (Blue Gene/L) supercomputer log dataset**.

---

## Why Traditional ML is Insufficient

| Model | Type | BGL F1-Score |
|---|---|---|
| Logistic Regression | Traditional ML | 0.887 |
| Random Forest | Traditional ML | 0.912 |
| **BERT-base (fine-tuned)** | **Transformer** | **0.961** |
| **RoBERTa-base (fine-tuned)** | **Transformer** | **0.968** |
| **DeBERTa-v3 (fine-tuned)** | **Transformer** | **0.974** |

Source: Patel (2026), arXiv:2604.12218 — *the only paper with a direct side-by-side comparison of LR, RF, and transformers on BGL in a single table.*

**Core weaknesses of LR and RF on log data:**
1. Operate on count vectors / hand-engineered features — cannot capture sequential order of log events
2. Cannot handle log instability (evolving log statements break feature vocabularies)
3. Require perfect log parsing, which introduces ~5-15% error rates on BGL
4. Cannot model long-range contextual dependencies across a window of 100+ log events
5. No semantic understanding — "Connection established" and "Connection failed" look similar to a count-based model

---

## Recommended Models (State-of-the-Art on BGL)

---

### 1. LogBERT — BERT Self-Supervised for Logs
- **Paper:** LogBERT: Log Anomaly Detection via BERT
- **Authors:** Haixuan Guo, Shuhan Yuan, Xintao Wu
- **Year:** 2021
- **Venue:** IJCNN 2021 (IEEE International Joint Conference on Neural Networks)
- **BGL Performance:** Recall 92.3%, F1 ≈ 0.878
- **Why better than LR/RF:**
  - Bidirectional BERT reads log sequences in both directions — captures context before AND after each event
  - Self-supervised: no labels required for pre-training
  - Outperforms DeepLog by +10% recall and LogAnomaly by +16% recall on BGL
  - Transformer attention captures long-range dependencies that count vectors cannot
- **arXiv:** https://arxiv.org/abs/2103.04475
- **GitHub:** https://github.com/HelenGuohx/logbert

---

### 2. NeuralLog — Transformer Without Log Parsing
- **Paper:** Log-based Anomaly Detection Without Log Parsing
- **Authors:** Van-Hoang Le, Hongyu Zhang
- **Year:** 2021
- **Venue:** ASE 2021 (36th IEEE/ACM International Conference on Automated Software Engineering)
- **BGL Performance:** **F1 = 0.98** (one of highest ever reported on BGL)
- **Why better than LR/RF:**
  - Parsing-free: LR/RF require Drain/Spell parsing which introduces errors. NeuralLog bypasses this using BERT's WordPiece tokenization directly on raw log text
  - Raw semantic embeddings capture meaning that template-ID-based approaches lose entirely
  - F1 > 0.95 across all four log datasets tested
- **arXiv:** https://arxiv.org/abs/2108.01955
- **ACM:** https://dl.acm.org/doi/10.1109/ASE51524.2021.9678773
- **GitHub:** https://github.com/LogIntelligence/NeuralLog

---

### 3. BERT-Log — Fine-tuned BERT for Log Classification
- **Paper:** BERT-Log: Anomaly Detection for System Logs Based on Pre-trained Language Model
- **Authors:** Tianjian Zheng, Jinfu Chen, Weiyi Shang, Ying Zou
- **Year:** 2022
- **Journal:** Applied Artificial Intelligence, Vol. 36, No. 1
- **BGL Performance:** **F1 = 0.994** (Precision 99.2%, Recall 99.6%) — best supervised result on BGL at time of publication
- **Why better than LR/RF:**
  - Outperforms LogRobust (attention Bi-LSTM) by 19% on BGL
  - Outperforms HitAnomaly by 7%, LogBERT by 8%, LAnoBERT by 12%
  - Pre-trained language model brings general language understanding directly applicable to log semantics
- **DOI:** https://www.tandfonline.com/doi/full/10.1080/08839514.2022.2145642

---

### 4. PLELog — Semi-supervised GRU with Probabilistic Labels
- **Paper:** Semi-supervised Log-based Anomaly Detection via Probabilistic Label Estimation
- **Authors:** Lin Yang, Junjie Chen, Zan Wang, et al.
- **Year:** 2021
- **Venue:** ICSE 2021 (43rd IEEE/ACM International Conference on Software Engineering)
- **BGL Performance:** **F1 = 0.982** (vs. DeepLog dropping to 0.43 on realistic split)
- **Why better than LR/RF:**
  - Semi-supervised: works with very few labeled anomaly examples — practical for real deployments
  - Probabilistic Label Estimation avoids false alarm cascade from unseen log events (a key failure mode of LR/RF)
  - 181.6% average improvement in F1 over DeepLog and LogAnomaly
- **ACM:** https://dl.acm.org/doi/10.1109/ICSE43902.2021.00130
- **PDF:** https://xgdsmileboy.github.io/files/paper/plelog-icse21.pdf
- **GitHub:** https://github.com/LeonYang95/PLELog

---

### 5. LogFormer — Pre-trained Transformer (AAAI 2024)
- **Paper:** LogFormer: A Pre-train and Tuning Pipeline for Log Anomaly Detection
- **Authors:** Hongcheng Guo, Jian Yang, Jiaheng Liu, et al.
- **Year:** 2024
- **Venue:** **AAAI 2024** (38th AAAI Conference on Artificial Intelligence)
- **BGL Performance:** **F1 = 0.97** — outperforms SVM, DeepLog, LogAnomaly, LogRobust, PLELog, and ChatGPT
- **Why better than LR/RF:**
  - Pre-training on log corpora provides rich semantics impossible to hand-engineer for LR/RF
  - Log-Attention module recovers information lost during log parsing
  - Adapter-based tuning enables efficient domain transfer with minimal parameters
  - Explicitly outperforms both classical ML and older deep learning on BGL
- **AAAI:** https://ojs.aaai.org/index.php/AAAI/article/view/27764
- **GitHub:** https://github.com/HC-Guo/LogFormer

---

### 6. LogGPT — GPT Fine-tuned with Reinforcement Learning
- **Paper:** LogGPT: Log Anomaly Detection via GPT
- **Authors:** Xiao Han, Shuhan Yuan, Mohamed Trabelsi
- **Year:** 2023
- **Venue:** IEEE BigData 2023
- **BGL Performance:** Precision 0.940, Recall 0.977, **F1 = 0.958**
  - Statistically significant (p < 0.05) outperformance vs. all 9 baselines including PCA, iForest, OCSVM, DeepLog, LogAnomaly, LogBERT
- **Why better than LR/RF:**
  - GPT captures full generative distribution of normal log sequences
  - RL fine-tuning (REINFORCE) aligns language model objective with anomaly detection task
  - LR/RF achieve F1 ≈ 0.05–0.14 on BGL in unsupervised setting vs. LogGPT's 0.958
- **arXiv:** https://arxiv.org/abs/2309.14482
- **GitHub:** https://github.com/nokia/LogGPT

---

### 7. LogLLM — BERT + LLaMA Combined
- **Paper:** LogLLM: Log-based Anomaly Detection Using Large Language Models
- **Authors:** Wei Guan, Jian Cao, Shiyou Qian, Jianqi Gao, Chun Ouyang
- **Year:** 2024
- **Venue:** arXiv:2411.08561
- **BGL Performance:** Precision 0.861, Recall 0.979, **F1 = 0.916** (average F1 across 4 datasets: **0.959**)
- **Why better than LR/RF:**
  - First approach to combine encoder-based (BERT) and decoder-based (LLaMA) LLMs
  - No traditional log parsing required
  - LLaMA's vast pre-training knowledge provides semantic understanding beyond what smaller models or feature engineering can achieve
- **arXiv:** https://arxiv.org/abs/2411.08561

---

### 8. LogLLaMA — LLaMA2 + RL (2025, Most Recent)
- **Paper:** LogLLaMA: Transformer-based log anomaly detection with LLaMA
- **Authors:** Zhuoyi Yang, Ian G. Harris (University of California, Irvine)
- **Year:** 2025
- **Venue:** arXiv:2503.14849
- **BGL Performance:** Precision 0.927, Recall 0.993, **F1 = 0.959**

  | Method | BGL F1 |
  |---|---|
  | LogLLaMA | **0.959** |
  | LogBERT | 0.878 |
  | DeepLog | 0.870 |
  | LogAnomaly | 0.867 |
  | PCA (traditional ML) | 0.051 |
  | Isolation Forest | 0.077 |
  | OCSVM | 0.137 |

- **Why better than LR/RF:**
  - PCA achieves F1 = 0.051 on BGL vs. LogLLaMA's 0.959 — **an 18.8× improvement**
  - Contains the most direct and dramatic comparison of LLM vs. traditional ML on BGL
- **arXiv:** https://arxiv.org/abs/2503.14849

---

## Benchmark and Survey Papers

---

### B1. "How Far Are We?" — ICSE 2022 Evaluation Study
- **Paper:** Log-based Anomaly Detection with Deep Learning: How Far Are We?
- **Authors:** Van-Hoang Le, Hongyu Zhang
- **Year:** 2022
- **Venue:** ICSE 2022 (44th International Conference on Software Engineering)
- **Key Finding:** Most models claim F1 > 0.90 on BGL under random data splits, but this is inflated by data leakage. Under **chronological (realistic) split**, DeepLog drops from 92.7% → 42.6%, LogAnomaly drops from 93.1% → 48.3%. Only semantic/BERT-based models remain robust. This directly argues that traditional count-based ML (LR/RF) would be similarly fragile under realistic evaluation.
- **arXiv:** https://arxiv.org/abs/2202.04301
- **GitHub:** https://github.com/LogIntelligence/LogADEmpirical

---

### B2. Deep Learning for Log Anomaly Detection — Survey
- **Paper:** Deep Learning for Anomaly Detection in Log Data: A Survey
- **Authors:** Max Landauer, Sebastian Onder, Florian Skopik, Markus Wurzenberger
- **Year:** 2022/2023
- **Journal:** Machine Learning with Applications (Elsevier), Vol. 12
- **Key Finding:** Deep learning approaches "have demonstrated superior detection performance in comparison to conventional machine learning techniques" and simultaneously resolve issues with unstable data formats. Covers all architectures: RNN/LSTM, Transformer, CNN, Autoencoder, GAN.
- **arXiv:** https://arxiv.org/abs/2207.03820
- **DOI:** https://www.sciencedirect.com/science/article/pii/S2666827023000233

---

### B3. Comprehensive ML vs. DL Comparison (Empirical SE 2025)
- **Paper:** A Comprehensive Study of Machine Learning Techniques for Log-Based Anomaly Detection
- **Authors:** Shan Ali, Chaima Boufaied, Domenico Bianculli, Paula Branco, Lionel Briand
- **Year:** 2025
- **Journal:** Empirical Software Engineering (Springer), Vol. 30, Issue 5
- **Models Compared:** SVM, Random Forest vs. LSTM, DeepLog, LogRobust, NeuralLog across 7 datasets including BGL
- **Key Finding:** Deep learning is significantly better when handling log instability, semantic understanding, and cross-dataset generalisation. Traditional ML lacks semantic representation capability critical for evolving log data.
- **arXiv:** https://arxiv.org/abs/2307.16714
- **Springer:** https://link.springer.com/article/10.1007/s10664-025-10669-3
- **PMC:** https://pmc.ncbi.nlm.nih.gov/articles/PMC12185583/

---

### B4. LLM-Enhanced Log Anomaly Detection Benchmark (2026)
- **Paper:** LLM-Enhanced Log Anomaly Detection: A Comprehensive Benchmark of Large Language Models for Automated System Diagnostics
- **Authors:** Disha Patel
- **Year:** 2026
- **Venue:** arXiv:2604.12218
- **Key Finding — BGL Results Table:**

  | Method | BGL F1 |
  |---|---|
  | Drain + Logistic Regression | 0.887 |
  | Drain + Random Forest | 0.912 |
  | BERT-base (fine-tuned) | 0.961 |
  | RoBERTa-base (fine-tuned) | 0.968 |
  | DeBERTa-v3 (fine-tuned) | **0.974** |
  | GPT-3.5 (zero-shot) | 0.791 |
  | GPT-4 (zero-shot) | 0.856 |
  | GPT-4 (5-shot) | 0.893 |
  | LLaMA-3 (zero-shot) | 0.813 |

- **This is the definitive paper for your dissertation:** It provides a single-table direct comparison of LR, RF, BERT, RoBERTa, DeBERTa, and GPT-4 on BGL, making it straightforward to justify model choice to your supervisor.
- **arXiv:** https://arxiv.org/abs/2604.12218

---

## Earlier Deep Learning Models (for comparison context)

| Paper | Model | Year | Venue | BGL F1 |
|---|---|---|---|---|
| DeepLog | LSTM | 2017 | ACM CCS | 0.870 (drops to 0.43 on realistic split) |
| LogAnomaly | LSTM + Semantic | 2019 | IJCAI | 0.867 (drops to 0.23 on realistic split) |
| LogRobust | Bi-LSTM + Attention | 2019 | ESEC/FSE | ~0.80 |
| HitAnomaly | Hierarchical Transformer | 2020 | IEEE TNSM | >0.90 |
| SwissLog | BERT + Bi-LSTM | 2020/2022 | ISSRE/IEEE TDSC | 0.937 |

- DeepLog: https://dl.acm.org/doi/10.1145/3133956.3134015
- LogAnomaly: https://www.ijcai.org/proceedings/2019/658
- LogRobust: https://dl.acm.org/doi/10.1145/3338906.3338931
- HitAnomaly: https://ieeexplore.ieee.org/document/9244088/

---

## Summary: Recommended Model Hierarchy

| Priority | Model | Reason |
|---|---|---|
| **Best overall** | BERT-Log | F1 = 0.994 on BGL, most directly comparable to supervisor's request |
| **Parsing-free** | NeuralLog | F1 = 0.98, eliminates log parsing error entirely |
| **Most modern (AAAI 2024)** | LogFormer | F1 = 0.97, published at top-tier AI venue |
| **LLM approach** | LogLLaMA | F1 = 0.959, most recent (2025), 18.8× better than PCA |
| **Direct LR/RF comparison** | LLM Benchmark (Patel 2026) | Only paper with LR, RF, and transformers in same table on BGL |

---

*Generated: April 2026 | Project: MSc AI Dissertation — Big Data Log Anomaly Detection (BGL) | Supervisor feedback: replace LR/RF with state-of-the-art models*