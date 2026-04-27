# Interpreting model performance (dashboard & `/performance`)

## Why you might see 1.000 for accuracy, F1, precision, or recall

1. **These numbers are from one saved holdout file** — `ml_pipeline/saved_models/eval_holdout.npz`, produced by `scripts/train_pipeline.py`. The UI recomputes metrics when the API starts so the table matches the **same** `model.predict()` path as live `/api/v1/predict` (not a “best possible” threshold fit on the test labels).

2. **Small *n* makes “perfect” scores common.**  
   If the holdout has only a few windows (e.g. tens of windows), a model can get **no errors on that slice**, so **accuracy = (TP+TN)/n = 1.000** and F1 = 1.000 even when the system is **not** perfect on the full 4.7M-line BGL corpus. This is a **point estimate on a small test set**, not a population guarantee.

3. **What to report in a dissertation**  
   - Always state **n** (number of test windows) next to the headline metrics.  
   - Report the **confusion table** (TP, FP, TN, FN) from the performance page or API — a reader can then see that, e.g., “1.0 accuracy” means 59/59 correct, not 1M/1M.  
   - Prefer **AUC-ROC** (ranking quality on the same holdout) as a complementary metric; it is often more informative when scores are well-separated.  
   - For **stronger claims**, increase the test split size, use **time-based** holdout (train on past, test on later logs), or **k-fold** / repeated evaluation — all require pipeline changes, not just the UI.

4. **Metrics source in the app**  
   - `holdout_eval` — live recomputation on `eval_holdout.npz` (intended).  
   - `training_metadata` or `offline_stub` — fallbacks if the holdout file is missing or the API is offline; treat numbers as **illustrative** until training is run and the API is healthy.

5. **API fields** (for your write-up or automation)  
   - `n_eval_samples`, `metric_note` — size caveat and text disclaimer.  
   - `tp`, `fp`, `tn`, `fn` — raw confusion counts.  
   - `metrics_source` — where the row came from.

## Related code

- `ml_pipeline/src/evaluator.py` — `precision_score`, `recall_score`, `f1_score`, `accuracy_score` on `y_pred = model.predict(X_test)`.  
- `ml_pipeline/api/app.py` — `/api/v1/metrics`, `/api/v1/models` expose the same evaluation block.

## Full-BGL “real” performance

The dashboard is wired for **inference and transparency on the training pipeline’s test split**, not a separate streaming benchmark over the entire BGL file. Stating “accuracy on BGL” for the whole LogHub download requires a **dedicated** evaluation design (sampling, labels, and compute) beyond this UI.
