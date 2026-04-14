# ML integration

The Rails app does not train models. It forwards normalized log payloads to your inference service and persists returned scores on `LogEntry` / `ModelMetric` as appropriate. Configure base URL and timeouts in the service object or credentials.

