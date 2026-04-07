class PerformanceController < ApplicationController
  def index
    @models_resp = MlApiService.get_models
    @models      = @models_resp["models"] || []
    @active      = @models.find { |m| m["is_active"] } || @models.first
    @trained_at  = @models_resp["trained_at"]

    # Best model highlight
    @best = @models.max_by { |m| m["f1_score"].to_f }

    # For ROC curve: generate representative curves for chart
    @roc_data = @models.map do |m|
      name = m["name"]
      auc  = m["auc_roc"].to_f
      # Approximate ROC curve from AUC using a beta distribution curve
      fpr_pts = (0..20).map { |i| i / 20.0 }
      tpr_pts = fpr_pts.map do |fpr|
        [[(fpr ** (1.0 / [auc * 2, 0.5].max)).round(3), 0.0].max, 1.0].min
      end
      { name: name, fpr: fpr_pts, tpr: tpr_pts, auc: auc }
    end

    # Dataset stats
    @dataset_info = {
      name:     "BGL (Blue Gene/L) Supercomputer Logs",
      source:   "LogHub — Lawrence Livermore National Laboratory",
      total:    4_747_963,
      anomalies: 348_460,
      normal:   4_399_503,
      imbalance_ratio: "1:12.6",
      window_size: 20,
      stride: 10,
    }
  end
end
