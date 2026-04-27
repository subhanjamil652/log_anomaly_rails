class DashboardController < ApplicationController
  def index
    @metrics      = MlApiService.get_metrics
    @models_resp  = MlApiService.get_models
    @models       = bert_log_models_only(@models_resp["models"] || [])
    @metrics_source = @models_resp["metrics_source"] || @metrics["metrics_source"]
    @active_model = @models.find { |m| m["is_active"] } || @models.first

    # Stats for stat cards
    @today_alerts    = AnomalyAlert.where("detected_at >= ?", Time.current.beginning_of_day).count
    @total_alerts   = AnomalyAlert.count
    # Log lines ingested from "Analyse logs" today (updates after each run)
    @log_lines_today = LogEntry.where("processed_at >= ?", Time.current.beginning_of_day)
                               .where(source: "uploaded").count

    # Recent alerts for the table
    @recent_alerts = AnomalyAlert.order(detected_at: :desc).limit(8)

    # Hourly alert counts for the last 24 h (for the chart)
    @hourly_counts = (0..23).map do |h|
      t_start = h.hours.ago.beginning_of_hour
      t_end   = t_start + 1.hour
      AnomalyAlert.where(detected_at: t_start..t_end).count
    end.reverse

    @hourly_labels = (0..23).map { |h| h.hours.ago.strftime("%H:%M") }.reverse

    # Severity distribution
    @severity_counts = {
      "critical" => AnomalyAlert.where(severity: "critical").count,
      "high"     => AnomalyAlert.where(severity: "high").count,
      "medium"   => AnomalyAlert.where(severity: "medium").count,
      "low"      => AnomalyAlert.where(severity: "low").count
    }
  end

  # JSON for live stat cards + charts (poll from dashboard view)
  def snapshot
    m = MlApiService.get_metrics
    mod = MlApiService.get_models
    today = Time.current.beginning_of_day
    hourly_counts = (0..23).map do |h|
      t_start = h.hours.ago.beginning_of_hour
      t_end   = t_start + 1.hour
      AnomalyAlert.where(detected_at: t_start..t_end).count
    end.reverse
    render json: {
      metrics: m,
      models: bert_log_models_only(mod["models"] || []),
      metrics_source: m["metrics_source"] || mod["metrics_source"],
      today_alerts: AnomalyAlert.where("detected_at >= ?", today).count,
      log_lines_today: LogEntry.where("processed_at >= ?", today).where(source: "uploaded").count,
      total_alerts: AnomalyAlert.count,
      hourly_counts: hourly_counts,
      severity: {
        critical: AnomalyAlert.where(severity: "critical").count,
        high: AnomalyAlert.where(severity: "high").count,
        medium: AnomalyAlert.where(severity: "medium").count,
        low: AnomalyAlert.where(severity: "low").count
      },
      updated_at: Time.current.iso8601
    }
  end
end
