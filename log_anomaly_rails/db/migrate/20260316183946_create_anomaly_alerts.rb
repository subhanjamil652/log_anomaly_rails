class CreateAnomalyAlerts < ActiveRecord::Migration[8.1]
  def change
    create_table :anomaly_alerts do |t|
      t.text :log_sequence
      t.boolean :is_anomaly
      t.float :confidence_score
      t.float :anomaly_score
      t.string :alert_model_name
      t.text :feature_importances
      t.datetime :detected_at
      t.string :status
      t.string :severity

      t.timestamps
    end
  end
end
