class AnalysisController < ApplicationController
  # Large pastes cannot live in the session cookie (4KB limit). Store the result
  # in Rails.cache and keep only a short key in the session.
  ANALYSIS_CACHE_TTL = 2.hours

  def index
    @sample_logs = sample_bgl_lines
    @metrics = MlApiService.get_metrics
  end

  def create
    raw_input = params[:log_input].to_s.strip
    if raw_input.empty? && params[:log_file].present?
      raw_input = params[:log_file].read.force_encoding("UTF-8")
    end

    lines = raw_input.split("\n").map(&:strip).reject(&:empty?)
    if lines.empty?
      redirect_to analysis_path, alert: "No log lines provided." and return
    end

    use_hf = ActiveModel::Type::Boolean.new.cast(params[:use_hf_semantics])
    result    = MlApiService.predict(lines, use_hf_semantics: use_hf)
    explain   = MlApiService.explain(lines.first(20))
    preds     = result["predictions"] || []
    summary   = result["summary"] || {}

    # Persist anomaly alerts
    preds.each do |pred|
      next unless pred["is_anomaly"]
      # Severity bands use raw P(anomaly) when the API provides it; display score is UI-scaled.
      raw = (pred["p_anomaly"] || pred["anomaly_score"]).to_f
      AnomalyAlert.create!(
        log_sequence:      lines[0, 5].join("\n"),
        is_anomaly:        true,
        confidence_score:  pred["confidence"].to_f,
        anomaly_score:     pred["anomaly_score"].to_f,
        alert_model_name:  pred["model"] || "BERT-Log",
        feature_importances: (explain["feature_importances"] || []).to_json,
        detected_at:       Time.current,
        status:            "new",
        severity:          score_to_severity(raw),
      )
    end

    # Persist log entries
    lines.first(50).each do |line|
      is_anom = preds.any? { |p| p["is_anomaly"] }
      LogEntry.create!(
        raw_content:   line,
        component:     extract_component(line),
        severity_level: extract_level(line),
        is_anomaly:    is_anom,
        anomaly_score: preds.first&.dig("anomaly_score").to_f,
        processed_at:  Time.current,
        source:        "uploaded",
      )
    end

    max_show = 2_000
    show_n = [lines.length, preds.length, max_show].min
    payload = {
      "lines" => lines.first(show_n),
      "preds" => preds.first(show_n),
      "summary" => summary,
      "explain" => explain
    }
    cache_key = "analysis_result:#{SecureRandom.hex(20)}"
    Rails.cache.write(cache_key, payload, expires_in: ANALYSIS_CACHE_TTL)
    session[:last_analysis_key] = cache_key
    session.delete(:last_analysis)
    redirect_to analysis_result_path("current"), status: :see_other
  end

  def show
    data = load_last_analysis_payload
    @lines   = data["lines"] || data[:lines] || []
    @preds   = data["preds"] || data[:preds] || []
    @summary = data["summary"] || data[:summary] || {}
    @explain = data["explain"] || data[:explain] || {}
    @n_anomaly = @preds.count { |p| p["is_anomaly"] }
    if @lines.empty?
      redirect_to analysis_path,
        notice: "No analysis data. Please submit logs. (Results expire after #{ANALYSIS_CACHE_TTL.inspect}.)"
    end
  end

  private

  def load_last_analysis_payload
    key = session[:last_analysis_key].presence
    if key
      cached = Rails.cache.read(key)
      return cached if cached.is_a?(Hash)
    end
    # Backward compatibility: old sessions stored a small hash in the cookie
    legacy = session[:last_analysis]
    return legacy if legacy.is_a?(Hash) && legacy["lines"].present?

    {}
  end

  def score_to_severity(score)
    case score
    when 0.85..1.0  then "critical"
    when 0.70..0.85 then "high"
    when 0.50..0.70 then "medium"
    else "low"
    end
  end

  def extract_component(line)
    parts = line.split(" ")
    parts.length >= 8 ? parts[7] : "UNKNOWN"
  end

  def extract_level(line)
    %w[FATAL SEVERE ERROR WARNING INFO].each do |lvl|
      return lvl if line.include?(lvl)
    end
    "UNKNOWN"
  end

  def sample_bgl_lines
    [
      "- 1117838570 2005.06.03 15:02:50.548667 R06-M1-N04-J01 RAS KERNEL KERNEL ciod: failed to read message prefix on control stream 5: Connection reset by peer",
      "- 1117838572 2005.06.03 15:02:52.113030 R06-M1-N04-J01 RAS APP APP ciod: MPI task 0 exited with status 1; aborting job",
      "FATAL 1117906684 2005.06.04 10:51:24.913855 R23-M0-N09-C0-J01 RAS KERNEL KERNEL data bus error",
      "- 1117907000 2005.06.04 10:57:30.000000 R01-M0-N02-J00 RAS APP APP program loaded",
      "SEVERE 1117908000 2005.06.04 11:00:00.000000 R05-M1-N08-J00 RAS KERNEL KERNEL uncorrectable ECC memory error at address 0x7f9a",
      "- 1117909000 2005.06.04 11:17:30.000000 R12-M2-N01-J00 RAS APP APP job 44321 completed successfully"
    ]
  end
end
