# Internal API surface

Controllers under `app/controllers` expose HTML views today. JSON endpoints can be added for ingestion webhooks; keep request validation and rate limits at the edge before calling `MlApiService`.

