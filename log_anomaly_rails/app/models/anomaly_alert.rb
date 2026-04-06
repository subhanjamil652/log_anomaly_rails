class AnomalyAlert < ApplicationRecord
  validates :anomaly_score, numericality: { in: 0.0..1.0 }, allow_nil: true
  validates :severity, inclusion: { in: %w[critical high medium low] }, allow_nil: true
  validates :status,   inclusion: { in: %w[new reviewed resolved] },    allow_nil: true

  scope :today,      -> { where("detected_at >= ?", Time.current.beginning_of_day) }
  scope :unresolved, -> { where.not(status: "resolved") }
  scope :by_severity,-> { order(Arel.sql("CASE severity WHEN 'critical' THEN 1 WHEN 'high' THEN 2 WHEN 'medium' THEN 3 ELSE 4 END")) }
end
