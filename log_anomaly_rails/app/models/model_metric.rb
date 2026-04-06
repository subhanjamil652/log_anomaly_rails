class ModelMetric < ApplicationRecord
  scope :active, -> { where(is_active: true) }
  scope :best,   -> { order(f1_score: :desc).first }
end
