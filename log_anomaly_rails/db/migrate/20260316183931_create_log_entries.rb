class CreateLogEntries < ActiveRecord::Migration[8.1]
  def change
    create_table :log_entries do |t|
      t.text :raw_content
      t.string :template
      t.string :component
      t.string :severity_level
      t.boolean :is_anomaly
      t.float :anomaly_score
      t.datetime :processed_at
      t.string :source

      t.timestamps
    end
  end
end
