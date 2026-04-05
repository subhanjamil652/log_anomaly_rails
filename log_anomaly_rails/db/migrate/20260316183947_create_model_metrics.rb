class CreateModelMetrics < ActiveRecord::Migration[8.1]
  def change
    create_table :model_metrics do |t|
      t.string :metric_model_name
      t.float :f1_score
      t.float :precision_score
      t.float :recall_score
      t.float :auc_roc
      t.float :accuracy
      t.float :false_positive_rate
      t.float :detection_latency_ms
      t.integer :training_samples
      t.integer :test_samples
      t.datetime :trained_at
      t.boolean :is_active
      t.string :model_type

      t.timestamps
    end
  end
end
