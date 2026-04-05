Rails.application.routes.draw do
  root "dashboard#index"

  get  "dashboard", to: "dashboard#index", as: :dashboard

  get  "analysis",       to: "analysis#index"
  post "analysis",       to: "analysis#create"
  get  "analysis/:id",   to: "analysis#show",   as: :analysis_result

  get  "alerts",         to: "alerts#index",    as: :alerts
  get  "alerts/:id",     to: "alerts#show",     as: :alert
  patch "alerts/:id",    to: "alerts#update"
  patch "alerts/:id/resolve", to: "alerts#resolve", as: :resolve_alert
  patch "alerts/:id/review",  to: "alerts#review",  as: :review_alert
  delete "alerts/:id",   to: "alerts#destroy"

  get  "performance",    to: "performance#index", as: :performance

  get  "monitor",        to: "monitor#index",   as: :monitor
  get  "monitor/stream", to: "monitor#stream",  as: :monitor_stream

  get "up" => "rails/health#show", as: :rails_health_check
end
