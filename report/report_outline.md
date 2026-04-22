# Report Outline — CENG 454 Group 2
## Parameter-Efficient Fine-Tuning Ensembles for Sentiment Analysis

**Format**: PDF, ≤8 pages (excl. cover + bibliography), Times New Roman 11pt, 1.0 spacing, 2.5cm margins

---

## Cover Page (not counted in 8 pages)
- Çankaya University / Faculty of Engineering / Computer Engineering
- CENG 454 — Artificial Intelligence and Data Science
- "Parameter-Efficient Fine-Tuning Ensembles for Sentiment Analysis"
- Group 2: Mustafa Onur Bayram (210401018), Batuhan Türkaslan (210401052), Batuhan Bilecen (210401002), Efekan Kandak (210401029)
- Submission Date: May 2026

---

## 1. Introduction and Motivation (~0.75 page)
- Problem: Sentiment analysis is critical for business/social media; transformers are powerful but expensive to fine-tune
- Motivation: LoRA enables cheap fine-tuning; ensembles boost accuracy — can we combine them?
- Research question: Can a stacking ensemble of LoRA-adapted models match/exceed full fine-tuning?
- Contribution summary: We show PEFT ensembles offer better accuracy at lower compute cost

## 2. Background and Related Work (~1.25 pages)
- Transformer architectures (BERT, RoBERTa, DeBERTa) — cite original papers
- Parameter-efficient fine-tuning: LoRA (Hu et al., 2022), adapters, prefix tuning
- Ensemble methods in NLP: bagging, boosting, stacking
- Sentiment analysis benchmarks and prior results on IMDB
- **Must cite ≥10 sources in IEEE format**

### Suggested References
1. Devlin et al. (2019) — BERT
2. Liu et al. (2019) — RoBERTa
3. He et al. (2021) — DeBERTa / DeBERTa-v3
4. Hu et al. (2022) — LoRA
5. Maas et al. (2011) — IMDB dataset
6. Go et al. (2009) — Sentiment140 dataset
7. Wolpert (1992) — Stacked Generalization
8. Sun et al. (2019) — Fine-tuning BERT for text classification
9. Ding et al. (2023) — Parameter-efficient fine-tuning survey
10. Wang et al. (2019) — GLUE / SuperGLUE benchmarks
11+ Additional relevant papers as needed

## 3. Dataset and Preprocessing (~0.75 page)
- IMDB: source, size (50k), binary labels, pre-split, text characteristics
- Sentiment140: source, original size (1.6M), subsampled to 50k, justification
- Preprocessing: tokenization with model-specific tokenizers, max_length=256, padding/truncation
- Train/val split strategy: 20k train + 5k validation (for ensemble) + 25k test
- Table: Dataset statistics (avg length, vocab size, class distribution)

## 4. Proposed Approach / Implementation (~1.5 pages)
- Overall pipeline diagram (Figure 1)
- Baseline: full fine-tuning RoBERTa (all 125M params)
- LoRA setup: r=8, α=16, target Q/V modules, trainable param counts
- Training details: epochs, batch size, learning rate, optimizer, fp16
- Stacking ensemble: architecture, meta-learner (LogReg), data leakage prevention
- Figure 2: Ensemble architecture diagram

## 5. Evaluation and Results (~1.5 pages)
- Table 1: Main results (Accuracy, F1, Training Time, Trainable Params) for all models × datasets
- Table 2: Parameter efficiency comparison
- Figure 3: Bar chart comparing accuracy across models
- Figure 4: Training time comparison
- Figure 5: Confusion matrices (2×2 grid)
- Sentiment140 cross-domain results
- Statistical observations

## 6. Discussion and Limitations (~1.0 page)
- Interpret results: did ensemble beat individual models? By how much?
- LoRA efficiency: parameter savings, time savings, memory savings
- Error analysis: what types of text are hard? (sarcasm, mixed sentiment, short text)
- Limitations: Colab constraints, subsample size, binary-only classification, English-only
- What worked vs. what didn't — be honest

## 7. Conclusion and Future Work (~0.5 page)
- Summary of findings
- Answer the research question directly
- Future work: more models in ensemble, multi-class sentiment, multilingual, larger datasets, other PEFT methods

## 8. References (not counted in 8 pages)
- IEEE citation format
- Minimum 10 sources
- Include all papers cited in Background section

---

## Page Budget
| Section | Target Pages |
|---|---|
| Introduction | 0.75 |
| Background | 1.25 |
| Dataset | 0.75 |
| Approach | 1.50 |
| Results | 1.50 |
| Discussion | 1.00 |
| Conclusion | 0.50 |
| **Total** | **7.25** |
| **Buffer** | **0.75** |

## Tips for Staying Under 8 Pages
- Use tables instead of prose for numerical results
- Keep figures compact (2-column layout where possible)
- Avoid repeating information across sections
- Use concise academic language
- Put implementation details in the notebook, not the report
