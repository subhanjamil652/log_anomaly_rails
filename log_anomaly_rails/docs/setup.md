# Local setup

1. `bundle install`
2. `yarn install` (if using the asset pipeline helpers in `package.json`)
3. Copy and adjust environment variables; ensure `config/master.key` exists for encrypted credentials.
4. `bin/rails db:prepare`
5. `bin/dev` or `bin/rails server`

