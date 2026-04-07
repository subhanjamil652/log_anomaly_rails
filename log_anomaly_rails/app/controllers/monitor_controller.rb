class MonitorController < ApplicationController
  include ActionController::Live

  def index
    @n_logs      = 60
    @anomaly_rate = 0.08
  end

  def stream
    response.headers["Content-Type"]  = "text/event-stream"
    response.headers["Cache-Control"] = "no-cache"
    response.headers["X-Accel-Buffering"] = "no"

    n_logs      = (params[:n_logs] || 60).to_i.clamp(10, 200)
    anomaly_rate = (params[:anomaly_rate] || 0.08).to_f.clamp(0.01, 0.5)

    begin
      result = MlApiService.simulate(n_logs: n_logs, anomaly_rate: anomaly_rate)
      logs   = result["logs"] || []
      summary = result["summary"] || {}

      logs.each_with_index do |log, idx|
        data = {
          index:        idx + 1,
          line:         log["line"],
          is_anomaly:   log["is_anomaly"],
          anomaly_score: log["anomaly_score"],
          confidence:   log["confidence"],
          component:    log["component"],
          level:        log["level"],
          progress:     ((idx + 1).to_f / logs.size * 100).round(1),
        }
        response.stream.write("data: #{data.to_json}\n\n")
        sleep 0.15
      end

      # Final summary event
      response.stream.write("event: complete\ndata: #{summary.to_json}\n\n")
    rescue ActionController::Live::ClientDisconnected
      # client closed connection — stop gracefully
    ensure
      response.stream.close
    end
  end
end
