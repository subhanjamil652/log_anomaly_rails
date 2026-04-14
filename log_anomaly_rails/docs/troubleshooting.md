# Troubleshooting

- **ML timeouts**: Check inference service health and timeouts inside `MlApiService`.
- **Missing assets**: Run `yarn build` or `bin/dev` so CSS builds land in `app/assets/builds`.
- **DB errors**: Confirm `db/schema.rb` matches migrations after pulling new code.

