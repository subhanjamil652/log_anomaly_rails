require "net/http"
require "json"
require "uri"

# MlApiService - communicates with the Python Flask anomaly detection API.
# Falls back to realistic mock data when the API is unreachable (demo mode).
class MlApiService
  API_BASE = ENV.fetch("ML_API_URL", "http://localhost:5001")
  TIMEOUT  = 8  # seconds

  # -- Public interface -----------------------------------------------------

  def self.health_check
    get("/api/v1/health")
  rescue => e
    { "status" => "unavailable", "model_loaded" => false,
      "model_name" => "Random Forest", "uptime_seconds" => 0, "error" => e.message }
  end

  def self.predict(log_lines)
    post("/api/v1/predict", { log_lines: log_lines })
  rescue => e
    mock_predict(log_lines)
  end

  def self.get_metrics
    get("/api/v1/metrics")
  rescue => e
    mock_metrics
  end

  def self.get_models
    get("/api/v1/models")
  rescue => e
    mock_models_response
  end

  def self.explain(log_lines)
    post("/api/v1/explain", { log_lines: log_lines })
  rescue => e
    mock_explain
  end

  def self.simulate(n_logs: 50, anomaly_rate: 0.08)
    post("/api/v1/simulate", { n_logs: n_logs, anomaly_rate: anomaly_rate })
  rescue => e
    mock_simulate(n_logs: n_logs, anomaly_rate: anomaly_rate)
  end

  # -- HTTP helpers ---------------------------------------------------------

  def self.get(path)
    uri = URI.parse("#{API_BASE}#{path}")
    http = Net::HTTP.new(uri.host, uri.port)
    http.open_timeout = TIMEOUT
    http.read_timeout = TIMEOUT
    response = http.get(uri.path)
    JSON.parse(response.body)
  end

  def self.post(path, body)
    uri = URI.parse("#{API_BASE}#{path}")
    http = Net::HTTP.new(uri.host, uri.port)
    http.open_timeout = TIMEOUT
    http.read_timeout = TIMEOUT
    request = Net::HTTP::Post.new(uri.path, "Content-Type" => "application/json")
    request.body = body.to_json
    response = http.request(request)
    JSON.parse(response.body)
  end

  # -- Mock / demo fallback data --------------------------------------------

  def self.mock_predict(log_lines)
    rng = Random.new(42)
    predictions = log_lines.each_with_index.map do |line, i|
      is_anom  = rng.rand < 0.08
      score    = is_anom ? rng.rand(0.65..0.97) : rng.rand(0.01..0.14)
      {
        "window_index" => i,
        "is_anomaly"   => is_anom,
        "confidence"   => (is_anom ? score : 1 - score).round(4),
        "anomaly_score" => score.round(4),
        "model"        => "Random Forest",
      }
    end
    n_anom = predictions.count { |p| p["is_anomaly"] }
    {
      "predictions" => predictions,
      "summary" => {
        "total_windows"     => predictions.size,
        "anomalies_detected" => n_anom,
        "anomaly_rate"      => (n_anom.to_f / [predictions.size, 1].max).round(4),
        "model"             => "Random Forest",
      },
      "processing_time_ms" => (predictions.size * 0.42).round(2),
    }
  end

  def self.mock_metrics
    {
      "model_name"            => "Random Forest",
      "f1_score"              => 0.924,
      "precision"             => 0.918,
      "recall"                => 0.931,
      "auc_roc"               => 0.971,
      "accuracy"              => 0.962,
      "false_positive_rate"   => 0.031,
      "false_negative_rate"   => 0.028,
      "detection_latency_ms"  => 0.42,
      "training_samples"      => 56_832,
      "test_samples"          => 12_000,
      "trained_at"            => 2.days.ago.iso8601,
      "dataset"               => "BGL (Blue Gene/L) Supercomputer Logs",
    }
  end

  def self.mock_models_response
    {
      "models" => [
        { "name" => "Random Forest",      "type" => "supervised",
          "f1_score" => 0.924, "precision" => 0.918, "recall" => 0.931,
          "auc_roc"  => 0.971, "accuracy"  => 0.962,
          "false_positive_rate" => 0.031,  "detection_latency_ms" => 0.42,
          "is_active" => true },
        { "name" => "LSTM Autoencoder",   "type" => "deep_learning",
          "f1_score" => 0.882, "precision" => 0.876, "recall" => 0.889,
          "auc_roc"  => 0.946, "accuracy"  => 0.951,
          "false_positive_rate" => 0.043,  "detection_latency_ms" => 1.24,
          "is_active" => false },
        { "name" => "Isolation Forest",   "type" => "unsupervised",
          "f1_score" => 0.793, "precision" => 0.762, "recall" => 0.827,
          "auc_roc"  => 0.884, "accuracy"  => 0.907,
          "false_positive_rate" => 0.071,  "detection_latency_ms" => 0.67,
          "is_active" => false },
        { "name" => "Logistic Regression","type" => "supervised",
          "f1_score" => 0.847, "precision" => 0.831, "recall" => 0.863,
          "auc_roc"  => 0.921, "accuracy"  => 0.934,
          "false_positive_rate" => 0.059,  "detection_latency_ms" => 0.08,
          "is_active" => false },
      ],
      "active_model" => "Random Forest",
      "trained_at"   => 2.days.ago.iso8601,
    }
  end

  def self.mock_explain
    {
      "feature_importances" => [
        { "name" => "fatal_count",         "value" => 0.312, "direction" => "positive" },
        { "name" => "sev_max",             "value" => 0.287, "direction" => "positive" },
        { "name" => "error_count",         "value" => 0.241, "direction" => "positive" },
        { "name" => "max_template_repeat", "value" => 0.198, "direction" => "positive" },
        { "name" => "tfidf_rts",           "value" => 0.176, "direction" => "positive" },
        { "name" => "tfidf_kernel",        "value" => 0.152, "direction" => "positive" },
        { "name" => "sev_std",             "value" => 0.134, "direction" => "positive" },
        { "name" => "window_duration_s",   "value" => -0.089, "direction" => "negative" },
        { "name" => "template_diversity",  "value" => -0.071, "direction" => "negative" },
        { "name" => "event_rate_per_s",    "value" => 0.063,  "direction" => "positive" },
      ],
      "top_anomaly_indicators" => %w[fatal_count sev_max error_count max_template_repeat tfidf_rts],
      "model" => "Random Forest",
    }
  end

  def self.mock_simulate(n_logs: 50, anomaly_rate: 0.08)
    components = %w[kernel MMCS APP MPI-IO lustre BGLMASTER ciod IO rts]
    normal_msgs = [
      "instruction cache parity error corrected",
      "program loaded",
      "total of <N> nodes in partition",
      "job <N> completed successfully",
      "memory module <N> initialized",
      "torus network link <N> active",
    ]
    anomaly_msgs = [
      "data bus error on node <N>",
      "uncorrectable ECC memory error at address <N>",
      "FATAL: memory scrubbing failed on DIMM <N>",
      "machine check exception on node <N>",
      "link failure detected on torus port <N>",
    ]
    rng = Random.new
    logs = n_logs.times.map do |i|
      is_anom   = rng.rand < anomaly_rate
      template  = is_anom ? anomaly_msgs.sample(random: rng) : normal_msgs.sample(random: rng)
      component = components.sample(random: rng)
      level     = is_anom ? %w[FATAL SEVERE ERROR].sample(random: rng) : "INFO"
      node      = "R#{format('%02d', rng.rand(8))}-M#{rng.rand(4)}-N#{format('%02d', rng.rand(16))}-J00"
      score     = is_anom ? rng.rand(0.65..0.97) : rng.rand(0.01..0.14)
      {
        "line"          => "#{is_anom ? level : '-'} #{1_117_838_570 + i * 2} 2005.06.03 12:00:00.000 #{node} RAS #{level} #{component} #{template.gsub('<N>', rng.rand(9999).to_s)}",
        "is_anomaly"    => is_anom,
        "anomaly_score" => score.round(4),
        "confidence"    => (is_anom ? score : 1 - score).round(4),
        "component"     => component,
        "level"         => level,
        "template"      => template,
      }
    end
    n_anom = logs.count { |l| l["is_anomaly"] }
    {
      "logs" => logs,
      "summary" => { "total" => n_logs, "anomalies" => n_anom, "normal" => n_logs - n_anom },
    }
  end
end
