# QuantX — project page

A single-page site for the QuantX Financial Document Intelligence RAG system,
built for the PowerMind Hackathon 2026 (2nd Rank · ₹50,000).

```
index.html              the whole page
assets/css/style.css    design system
assets/js/main.js       interactions (no dependencies)
assets/img/             screenshots, architecture diagrams, hackathon photos
.nojekyll               tell GitHub Pages to serve files as-is
images/, award ceramany/  original source images (not used by the page)
```

No build step, no framework, no npm install. Plain HTML, CSS and JS —
every path is relative, so the page works from any subdirectory.

## Preview locally

```bash
cd "FInancial Rag chatbot"
python -m http.server 8099
# open http://localhost:8099
```

## Deploy

A workflow at `.github/workflows/pages.yml` publishes this folder as the site
root. Enable it once:

**Repository → Settings → Pages → Build and deployment → Source: GitHub Actions**

Then push to `main`. The page goes live at:

```
https://yashingle-ai.github.io/-PS2-QuantX-Financial-Document-Intelligence-RAG/
```

### Alternative: deploy from a branch

If you would rather not use Actions, set **Source: Deploy from a branch →
main → / (root)**. The page is then served from the URL-encoded subpath:

```
https://yashingle-ai.github.io/-PS2-QuantX-Financial-Document-Intelligence-RAG/FInancial%20Rag%20chatbot/
```

The workflow route is recommended — the folder name contains a space, which
survives in the URL as `%20`.

## Notes on content

All figures on the page come from the repository or the running application —
nothing is invented. The screenshots showing `33 pages · 192 chunks · 30 tables`
and the `strong support` grounding badge are unmodified captures of the product.

The teammate profile screenshot from the source folder is deliberately not used:
it displays a personal email address.
