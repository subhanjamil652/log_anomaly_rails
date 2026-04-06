# This file is auto-generated from the current state of the database. Instead
# of editing this file, please use the migrations feature of Active Record to
# incrementally modify your database, and then regenerate this schema definition.
#
# This file is the source Rails uses to define your schema when running `bin/rails
# db:schema:load`. When creating a new database, `bin/rails db:schema:load` tends to
# be faster and is potentially less error prone than running all of your
# migrations from scratch. Old migrations may fail to apply correctly if those
# migrations use external dependencies or application code.
#
# It's strongly recommended that you check this file into your version control system.

ActiveRecord::Schema[8.1].define(version: 2026_03_16_183947) do
  create_table "anomaly_alerts", force: :cascade do |t|
    t.string "alert_model_name"
    t.float "anomaly_score"
    t.float "confidence_score"
    t.datetime "created_at", null: false
    t.datetime "detected_at"
    t.text "feature_importances"
    t.boolean "is_anomaly"
    t.text "log_sequence"
    t.string "severity"
    t.string "status"
    t.datetime "updated_at", null: false
  end

  create_table "log_entries", force: :cascade do |t|
    t.float "anomaly_score"
    t.string "component"
    t.datetime "created_at", null: false
    t.boolean "is_anomaly"
    t.datetime "processed_at"
    t.text "raw_content"
    t.string "severity_level"
    t.string "source"
    t.string "template"
    t.datetime "updated_at", null: false
  end

  create_table "model_metrics", force: :cascade do |t|
    t.float "accuracy"
    t.float "auc_roc"
    t.datetime "created_at", null: false
    t.float "detection_latency_ms"
    t.float "f1_score"
    t.float "false_positive_rate"
    t.boolean "is_active"
    t.string "metric_model_name"
    t.string "model_type"
    t.float "precision_score"
    t.float "recall_score"
    t.integer "test_samples"
    t.datetime "trained_at"
    t.integer "training_samples"
    t.datetime "updated_at", null: false
  end
end
