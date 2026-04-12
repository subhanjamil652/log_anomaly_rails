# Architecture

- **Web tier**: Rails 8 handles HTTP, authentication boundaries (extend as needed), and Hotwire views.
- **Data**: ActiveRecord models back log entries, alerts, and cached model metrics.
- **ML bridge**: `MlApiService` calls an external HTTP API for anomaly scores; failures degrade gracefully in the UI.
- **Jobs**: `ApplicationJob` is ready for Solid Queue or async score refresh workflows.

