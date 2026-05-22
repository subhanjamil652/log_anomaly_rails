module ApplicationHelper
  # BERT-Log can store *raw* P(anomaly) (~0.02) on older alerts; bars looked like ~2%.
  # The ML API now returns a display 0-1 scale; this rescues legacy DB rows in (0, 0.08].
  BERT_RAW_SCORE_CUTOFF = 0.08

  def bert_p_threshold_for_rescale
    x = ENV.fetch("ML_BERT_P_THRESHOLD", "0.0195").to_f
    x.positive? ? x : 0.0195
  end

  # Mirrors ml_pipeline.api.app._display_anomaly_score (legacy raw P in 0..0.08)
  def anomaly_score_for_ui(raw)
    v = raw.to_f
    # One-off old API: p/t hit exactly 1.0 for most alerts
    v = 0.91 if v >= 0.999
    return v if v > BERT_RAW_SCORE_CUTOFF
    return v if v <= 0
    t = bert_p_threshold_for_rescale
    return v if t >= 0.15
    span = 1.0 - t
    s = [[((v - t) / span), 0.0].max, 1.0].min
    w = s**0.42
    d = 0.62 + 0.30 * w
    format("%.4f", [[d, 0.92].min, 0.58].max).to_f
  end

  # 0–1 metrics (F1, AUC, …): fixed decimals so 1.000 is not shown as a bare "1"
  def ml_rate(v, precision: 3)
    return "—" if v.nil?
    format("%.#{precision}f", v.to_f)
  end

  def ml_pct(v, precision: 2)
    return "—" if v.nil?
    number_with_precision(v.to_f * 100, precision: precision) + "%"
  end
end
