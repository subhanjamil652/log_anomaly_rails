class AlertsController < ApplicationController
  before_action :set_alert, only: [:show, :update, :resolve, :review, :destroy]

  def index
    @alerts = AnomalyAlert.order(detected_at: :desc)
    @alerts = @alerts.where(severity: params[:severity]) if params[:severity].present?
    @alerts = @alerts.where(status: params[:status])     if params[:status].present?
    if params[:date_from].present?
      @alerts = @alerts.where("detected_at >= ?", params[:date_from].to_date.beginning_of_day)
    end
    @total_count    = @alerts.count
    @critical_count = AnomalyAlert.where(severity: "critical").count
    @new_count      = AnomalyAlert.where(status: "new").count
    @alerts = @alerts.limit(100)
  end

  def show
    @feature_importances = begin
      JSON.parse(@alert.feature_importances || "[]")
    rescue
      []
    end
  end

  def update
    @alert.update(status: params[:status]) if params[:status].present?
    redirect_to alerts_path, notice: "Alert updated."
  end

  def resolve
    @alert.update(status: "resolved")
    redirect_to alerts_path, notice: "Alert resolved."
  end

  def review
    @alert.update(status: "reviewed")
    redirect_to alerts_path, notice: "Alert marked as reviewed."
  end

  def destroy
    @alert.destroy
    redirect_to alerts_path, notice: "Alert deleted."
  end

  private

  def set_alert
    @alert = AnomalyAlert.find(params[:id])
  end
end
