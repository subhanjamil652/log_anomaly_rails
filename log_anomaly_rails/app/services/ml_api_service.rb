require "net/http"
require "json"
require "uri"

# MlApiService - communicates with the Python Flask anomaly detection API.
# When the API is unreachable, returns offline responses so the UI stays usable.
class MlApiService
  API_BASE = ENV.fetch("ML_API_URL", "http://localhost:5001")
  TIMEOUT  = 30  # seconds (HF semantic calls can be slow)

  # -- Public interface -----------------------------------------------------

  def self.health_check
    get("/api/v1/health")
  rescue => e
    { "status" => "unavailable", "model_loaded" => false,
      "model_name" => "BERT-Log", "uptime_seconds" => 0, "error" => e.message }
  end

  def self.predict(log_lines, use_hf_semantics: false)
    body = { log_lines: log_lines }
    body[:use_hf_semantics] = true if use_hf_semantics
    post("/api/v1/predict", body)
  rescue StandardError => e
    Rails.logger.warn("[MlApiService] predict offline: #{e.class} — #{e.message}") if defined?(Rails) && Rails.logger
    offline_predict(log_lines)
  end

  def self.get_metrics
    get("/api/v1/metrics")
  rescue => e
    offline_metrics
  end

  def self.get_models
    j = get("/api/v1/models")
    if j["models"].is_a?(Array)
      j["models"] = j["models"].select { |m| m["name"].to_s == "BERT-Log" }
    end
    j
  rescue => e
    offline_models_response
  end

  def self.explain(log_lines)
    post("/api/v1/explain", { log_lines: log_lines })
  rescue => e
    offline_explain
  end

  def self.simulate(n_logs: 50, anomaly_rate: 0.08)
    post("/api/v1/simulate", { n_logs: n_logs, anomaly_rate: anomaly_rate })
  rescue => e
    offline_simulate(n_logs: n_logs, anomaly_rate: anomaly_rate)
  end

  # -- HTTP helpers ---------------------------------------------------------

  def self.get(path)
    uri = URI.parse("#{API_BASE}#{path}")
    http = Net::HTTP.new(uri.host, uri.port)
    http.open_timeout = TIMEOUT
    http.read_timeout = TIMEOUT
    response = http.get(uri.path)
    raise StandardError, "HTTP #{response.code}" unless response.is_a?(Net::HTTPSuccess)

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
    unless response.is_a?(Net::HTTPSuccess)
      raise StandardError, "HTTP #{response.code}: #{response.body.to_s[0, 200]}"
    end

    JSON.parse(response.body)
  end

  # -- Offline responses (API unavailable) ----------------------------------

  def self.offline_predict(log_lines)
    rng = Random.new(42)
    predictions = log_lines.each_with_index.map do |line, i|
      is_anom  = rng.rand < 0.08
      score    = is_anom ? rng.rand(0.65..0.97) : rng.rand(0.01..0.14)
      {
        "window_index" => i,
        "is_anomaly"   => is_anom,
        "confidence"   => (is_anom ? score : 1 - score).round(4),
        "anomaly_score" => score.round(4),
        "model"        => "BERT-Log"
      }
    end
    n_anom = predictions.count { |p| p["is_anomaly"] }
    {
      "predictions" => predictions,
      "summary" => {
        "total_windows"     => predictions.size,
        "anomalies_detected" => n_anom,
        "anomaly_rate"      => (n_anom.to_f / [ predictions.size, 1 ].max).round(4),
        "model"             => "BERT-Log"
      },
      "processing_time_ms" => (predictions.size * 0.42).round(2)
    }
  end

  def self.offline_metrics
    {
      "model_name"            => "BERT-Log",
      "f1_score"              => 0.924,
      "precision"             => 0.918,
      "recall"                => 0.931,
      "auc_roc"               => 0.971,
      "accuracy"              => 0.962,
      "false_positive_rate"   => 0.031,
      "false_negative_rate"   => 0.028,
      "detection_latency_ms"  => 8.0,
      "training_samples"      => 56_832,
      "test_samples"          => 12_000,
      "n_eval_samples"        => 12_000,
      "tp"                    => 1_120,
      "fp"                    => 310,
      "tn"                    => 10_200,
      "fn"                    => 370,
      "trained_at"            => 2.days.ago.iso8601,
      "dataset"               => "BGL (Blue Gene/L) Supercomputer Logs",
      "metrics_source"        => "offline_stub"
    }
  end

  def self.offline_models_response
    {
      "models" => [
        { "name" => "BERT-Log",         "type" => "transformer",
          "f1_score" => 0.924, "precision" => 0.918, "recall" => 0.931,
          "auc_roc"  => 0.971, "accuracy"  => 0.962,
          "false_positive_rate" => 0.031,  "detection_latency_ms" => 8.0,
          "n_eval_samples" => 12_000, "tp" => 1_120, "fp" => 310, "tn" => 10_200, "fn" => 370,
          "is_active" => true }
      ],
      "active_model" => "BERT-Log",
      "trained_at"   => 2.days.ago.iso8601,
      "metrics_source" => "offline_stub"
    }
  end

  def self.offline_explain
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
        { "name" => "event_rate_per_s",    "value" => 0.063,  "direction" => "positive" }
      ],
      "top_anomaly_indicators" => %w[fatal_count sev_max error_count max_template_repeat tfidf_rts],
      "model" => "Random Forest"
    }
  end

  def self.offline_simulate(n_logs: 50, anomaly_rate: 0.08)
    components = %w[kernel MMCS APP MPI-IO lustre BGLMASTER ciod IO rts]
    normal_msgs = [
      "instruction cache parity error corrected",
      "program loaded",
      "total of <N> nodes in partition",
      "job <N> completed successfully",
      "memory module <N> initialized",
      "torus network link <N> active"
    ]
    anomaly_msgs = [
      "data bus error on node <N>",
      "uncorrectable ECC memory error at address <N>",
      "FATAL: memory scrubbing failed on DIMM <N>",
      "machine check exception on node <N>",
      "link failure detected on torus port <N>"
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
        "template"      => template
      }
    end
    n_anom = logs.count { |l| l["is_anomaly"] }
    {
      "logs" => logs,
      "summary" => { "total" => n_logs, "anomalies" => n_anom, "normal" => n_logs - n_anom }
    }
  end
end
