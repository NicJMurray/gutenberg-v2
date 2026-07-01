# Deployment

- Repo: `NicJMurray/gutenberg-v2`
- Purpose: Rare Word Explorer
- Canonical URL: `https://rare-words.njmurray.com`
- Cloudflare type: Worker with Static Assets
- Cloudflare Worker name: `gutenberg`
- Deploy method: GitHub Actions + Wrangler
- Local/manual deploy command: `npm run deploy`
- Wrangler command: `npm run build && wrangler deploy`

## GitHub Actions

This repo is a Worker project, so it keeps the Wrangler GitHub Actions workflow.

Pushing to `main` deploys through `.github/workflows/deploy.yml`.

GitHub repository secrets are required for the Cloudflare account ID and API token.

The build step copies `data/frequencies.csv` to `public/data/` with a hash-named filename and updates `public/data/manifest.json` before deploying.

## Note

The static Pages repos use Cloudflare Git integration and do not need GitHub Actions deploy workflows. This Worker repo is different and still uses Wrangler.
