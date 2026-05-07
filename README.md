# Gutenberg Rare Word Explorer

Cloudflare-hosted rare word explorer for `https://njmurray.com/gutenberg`.

The app is now a small static frontend plus a Cloudflare Worker route. It no longer uses Streamlit, Pandas, NLTK, Render, Python requirements, or a long-running server process.

## What It Does

- Loads `data/frequencies.csv` as a cached static asset.
- Parses the frequency list in a browser Web Worker.
- Processes uploaded `.txt` files entirely in the browser.
- Returns the rarest matched words by corpus frequency.
- Optionally shows counts, first-use context, and unmatched words.
- Exports selected rare words as CSV and unmatched words as TXT.

The former WordNet definitions and IPA pronunciations were removed because they required Python/NLTK data or external dictionary calls. Keeping the Cloudflare version static makes it faster, cheaper, and more reliable for this use case.

## CSV Size

The current frequency file is about 6.5 MB with 400,000 data rows. That is fine for Cloudflare: Cloudflare's current Pages and Workers Static Assets limit for a single static asset is 25 MiB.

If the frequency data grows beyond 25 MiB, move it to R2 or split it into hashed chunks.

Sources:
- Cloudflare Pages limits: https://developers.cloudflare.com/pages/platform/limits/
- Cloudflare Workers limits: https://developers.cloudflare.com/workers/platform/limits/

## Local Development

```bash
npm install
npm run dev
```

Wrangler will serve the app through the Worker path mount. Open the local URL shown by Wrangler and use `/gutenberg/`.

## Build

```bash
npm run build
```

The build script copies `data/frequencies.csv` into `public/gutenberg/data/` as a hash-named file and writes `public/gutenberg/data/manifest.json`. Generated data assets are ignored by Git.

## Deploy

```bash
npm run deploy
```

`wrangler.toml` is configured for:

```toml
[[routes]]
pattern = "njmurray.com/gutenberg*"
zone_name = "njmurray.com"
```

Make sure `njmurray.com` has a proxied DNS record in Cloudflare before deploying the route.

## njmurray.com Homepage Link

Update the homepage project nav from:

```html
<a href="https://gutenberg.njmurray.com">Rare Words</a>
```

to:

```html
<a href="/gutenberg/">Rare Words</a>
```

This repository's metadata and in-app nav already use `https://njmurray.com/gutenberg`.
