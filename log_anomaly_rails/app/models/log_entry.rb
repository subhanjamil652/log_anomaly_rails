class LogEntry < ApplicationRecord
  scope :anomalous, -> { where(is_anomaly: true) }
  scope :recent,    -> { order(processed_at: :desc) }
end
