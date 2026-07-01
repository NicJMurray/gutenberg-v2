# Gutenberg Rare Word Explorer

Cloudflare-hosted rare word explorer for `https://rare-words.njmurray.com`.

The app is a small static frontend plus a Cloudflare Worker route. It no longer uses Streamlit, Pandas, NLTK, Render, Python requirements, or a long-running server process.

## What it does

- Loads `data/frequencies.csv` as a cached static asset.
- Parses the frequency list in a browser Web Worker.
- Processes uploaded `.txt` files entirely in the browser.
- Returns the rarest matched words by corpus frequency.
- Optionally shows counts, first-use context, and unmatched words.
- Exports selected rare words as CSV and unmatched words as TXT.

The former WordNet definitions and IPA pronunciations were removed because they required Python/NLTK data or external dictionary calls. Keeping the Cloudflare version static makes it faster, cheaper, and more reliable for this use case.

## CSV size

The current frequency file is about 6.5 MB with 400,000 data rows. That is fine for Cloudflare's current Pages and Workers Static Assets single-file limits.

If the frequency data grows beyond the platform limit, move it to R2 or split it into hashed chunks.

## Local development

```bash
npm install
npm run dev
```

Wrangler will serve the app at the local URL shown by `wrangler dev`.

## Build

```bash
npm run build
```

The build script copies `data/frequencies.csv` into `public/data/` as a hash-named file and writes `public/data/manifest.json`. Generated data assets are ignored by Git.

## Deployment

This is a Cloudflare Worker with static assets, so it still deploys with Wrangler rather than the Pages Git integration used by the static Pages repos.

```bash
npm run deploy
```

Automatic deploys run through GitHub Actions on pushes to `main`.

`wrangler.toml` is configured for the custom domain route:

```toml
[[routes]]
pattern = "rare-words.njmurray.com"
custom_domain = true
```

See [DEPLOYMENT.md](DEPLOYMENT.md) for the deployment summary.
