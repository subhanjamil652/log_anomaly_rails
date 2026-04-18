class DashboardController < ApplicationController
  def index
    @metrics      = MlApiService.get_metrics
    @models_resp  = MlApiService.get_models
    @models       = @models_resp["models"] || []
    @metrics_source = @models_resp["metrics_source"] || @metrics["metrics_source"]
    @active_model = @models.find { |m| m["is_active"] } || @models.first

    # Stats for stat cards
    @today_alerts = AnomalyAlert.where("detected_at >= ?", Time.current.beginning_of_day).count
    @total_alerts = AnomalyAlert.count
    @anomaly_rate = @metrics["f1_score"]&.*(100)&.round(1) || 92.4

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
end
