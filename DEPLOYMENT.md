# Deployment

- Repo: `NicJMurray/gutenberg-v2`
- Purpose: Rare Word Explorer
- Canonical URL: `https://rare-words.njmurray.com`
- Cloudflare type: Worker with Static Assets
- Cloudflare Worker name: `gutenberg`
- Deploy command: `npm run deploy`
- Wrangler command: `npm run build && wrangler deploy`

## GitHub Actions

Pushing to `main` deploys through `.github/workflows/deploy.yml`.

Required repository secrets:

```text
CLOUDFLARE_API_TOKEN
CLOUDFLARE_ACCOUNT_ID
```

The build step copies `data/frequencies.csv` to `public/data/` with a hash-named filename and updates `public/data/manifest.json` before deploying.
