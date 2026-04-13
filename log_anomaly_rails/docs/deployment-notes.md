# Deployment notes

Production uses the stock Rails production defaults plus Kamal (`config/deploy.yml`). Ensure registry credentials and `RAILS_MASTER_KEY` are supplied via your secrets workflow, not committed plaintext.

