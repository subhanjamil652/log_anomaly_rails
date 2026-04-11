# Log Anomaly Rails

Rails dashboard and ingestion pipeline for log anomaly detection. The app stores log lines, surfaces ML-backed scores from a separate inference service, and provides monitoring views for operators.


## Prerequisites

- Ruby version in `.ruby-version`
- Node/Yarn for CSS/JS build tooling (see `package.json`)
- PostgreSQL (or adjust `config/database.yml` for your environment)
- Access to the ML scoring HTTP API expected by `MlApiService`


## Database

Run `bin/rails db:create db:migrate`. Seeds in `db/seeds.rb` optionally load demo log lines for UI development.


## Running

- `bin/dev` boots Rails with the Procfile-defined processes when available.
- Visit the dashboard and analysis routes defined in `config/routes.rb` after signing in (add auth as required).


## Contributing

Follow existing RuboCop rules (`.rubocop.yml`) and keep ML calls behind `MlApiService` so scoring backends can be swapped without churn in controllers.


## Features

- Dashboard summarizing recent anomalies
- Analysis views for drilling into log lines and scores
- Alerts lifecycle (index/show/update/destroy stubs)
- Live-ish monitor stream view and performance metrics page
- Bootstrap-themed UI via `application.bootstrap.scss`

