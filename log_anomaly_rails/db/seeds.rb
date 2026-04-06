# Seed data for BGL Log Anomaly Detection System
puts "Seeding database..."

# -- Model Metrics -------------------------------------------------------------
ModelMetric.delete_all

[
  { metric_model_name: "Random Forest",       model_type: "supervised",    f1_score: 0.924, precision_score: 0.918, recall_score: 0.931, auc_roc: 0.971, accuracy: 0.962, false_positive_rate: 0.031, detection_latency_ms: 0.42,  training_samples: 56832, test_samples: 12000, is_active: true  },
  { metric_model_name: "LSTM Autoencoder",    model_type: "deep_learning", f1_score: 0.882, precision_score: 0.876, recall_score: 0.889, auc_roc: 0.946, accuracy: 0.951, false_positive_rate: 0.043, detection_latency_ms: 1.24,  training_samples: 56832, test_samples: 12000, is_active: false },
  { metric_model_name: "Isolation Forest",    model_type: "unsupervised",  f1_score: 0.793, precision_score: 0.762, recall_score: 0.827, auc_roc: 0.884, accuracy: 0.907, false_positive_rate: 0.071, detection_latency_ms: 0.67,  training_samples: 56832, test_samples: 12000, is_active: false },
  { metric_model_name: "Logistic Regression", model_type: "supervised",    f1_score: 0.847, precision_score: 0.831, recall_score: 0.863, auc_roc: 0.921, accuracy: 0.934, false_positive_rate: 0.059, detection_latency_ms: 0.08,  training_samples: 56832, test_samples: 12000, is_active: false },
].each do |attrs|
  ModelMetric.create!(attrs.merge(trained_at: 2.days.ago))
end
puts "  [OK] #{ModelMetric.count} model metrics created"

# -- Anomaly Alerts ------------------------------------------------------------
AnomalyAlert.delete_all

anomaly_templates = [
  "data bus error on node R06-M1-N04",
  "uncorrectable ECC memory error at address 0x7f9a",
  "FATAL: memory scrubbing failed on DIMM 3",
  "machine check exception on node R23-M0-N09",
  "link failure detected on torus port 4",
  "node R05 failed to boot after 3 retries",
  "hardware watchdog timeout on core 12",
  "I/O forwarding layer unresponsive on node R12",
  "temperature threshold exceeded - 97 degrees",
  "network interface reset due to excessive errors",
]

components = %w[kernel MMCS rts ciod BGLMASTER IO lustre MPI-IO]
severities = { "critical" => 0.35, "high" => 0.30, "medium" => 0.25, "low" => 0.10 }
statuses   = { "new" => 0.50, "reviewed" => 0.30, "resolved" => 0.20 }

shap_features = [
  { "name" => "fatal_count",         "value" => 0.312, "direction" => "positive" },
  { "name" => "sev_max",             "value" => 0.287, "direction" => "positive" },
  { "name" => "error_count",         "value" => 0.241, "direction" => "positive" },
  { "name" => "max_template_repeat", "value" => 0.198, "direction" => "positive" },
  { "name" => "tfidf_rts",           "value" => 0.176, "direction" => "positive" },
  { "name" => "tfidf_kernel",        "value" => 0.152, "direction" => "positive" },
  { "name" => "sev_std",             "value" => 0.134, "direction" => "positive" },
  { "name" => "window_duration_s",   "value" => -0.089, "direction" => "negative" },
  { "name" => "template_diversity",  "value" => -0.071, "direction" => "negative" },
  { "name" => "event_rate_per_s",    "value" => 0.063, "direction" => "positive" },
]

def weighted_sample(hash)
  r = rand
  cumulative = 0
  hash.each do |k, prob|
    cumulative += prob
    return k if r < cumulative
  end
  hash.keys.last
end

60.times do |i|
  sev   = weighted_sample(severities)
  score = case sev
          when "critical" then rand(0.85..0.98)
          when "high"     then rand(0.70..0.85)
          when "medium"   then rand(0.50..0.70)
          else rand(0.30..0.50)
          end

  node = "R#{format('%02d', rand(8))}-M#{rand(4)}-N#{format('%02d', rand(16))}-J00"
  comp = components.sample
  tmpl = anomaly_templates.sample
  lvl  = sev == "critical" ? "FATAL" : sev == "high" ? "SEVERE" : "ERROR"

  log_seq = [
    "#{lvl} #{1_117_838_570 + rand(10000)} 2005.06.03 #{format('%02d', rand(24))}:#{format('%02d', rand(60))}:00.000 #{node} RAS #{lvl} #{comp} #{tmpl}",
    "- #{1_117_838_570 + rand(10000)} 2005.06.03 12:00:00.000 #{node} RAS INFO #{comp} system check in progress",
  ].join("\n")

  AnomalyAlert.create!(
    log_sequence:      log_seq,
    is_anomaly:        true,
    confidence_score:  score.round(4),
    anomaly_score:     score.round(4),
    alert_model_name:  "Random Forest",
    feature_importances: shap_features.to_json,
    detected_at:       rand(7).days.ago - rand(86400).seconds,
    status:            weighted_sample(statuses),
    severity:          sev,
  )
end
puts "  [OK] #{AnomalyAlert.count} anomaly alerts created"

# -- Log Entries ---------------------------------------------------------------
LogEntry.delete_all

normal_templates = [
  "instruction cache parity error corrected",
  "program loaded",
  "job <N> completed successfully",
  "memory module initialized",
  "torus network link active",
  "I/O node registered",
  "barrier reached by all tasks",
  "file opened for read",
]

80.times do |i|
  is_anom = rand < 0.08
  node_str = "R#{format('%02d', rand(8))}-M0-N00-J00"
  LogEntry.create!(
    raw_content:    "#{is_anom ? 'FATAL' : '-'} #{1_117_838_570 + i*2} 2005.06.03 12:00:00.000 #{node_str} RAS INFO #{components.sample} #{is_anom ? anomaly_templates.sample : normal_templates.sample}",
    component:      components.sample,
    severity_level: is_anom ? "FATAL" : "INFO",
    is_anomaly:     is_anom,
    anomaly_score:  is_anom ? rand(0.65..0.97).round(4) : rand(0.01..0.14).round(4),
    processed_at:   rand(3).days.ago,
    source:         "simulated",
  )
end
puts "  [OK] #{LogEntry.count} log entries created"

puts "\nSeed complete! Summary:"
puts "  ModelMetrics: #{ModelMetric.count}"
puts "  AnomalyAlerts: #{AnomalyAlert.count}"
puts "  LogEntries: #{LogEntry.count}"
