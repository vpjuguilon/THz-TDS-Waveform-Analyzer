# THz Waveform & Spectrum Bench — standalone build

This turns the tool into a real project you can
build, host online, or package as a shareable file. Everything runs entirely
in the browser (no backend, no server-side processing) — any data someone
loads into the tool stays on their own machine.

## Requirements

- [Node.js](https://nodejs.org) (LTS version, includes `npm`) — only needed on
  **your** machine for building/deploying. Visitors just need a browser.

## Option A — Host it online via GitHub (recommended for a lab tool)

This lets you keep editing the code in the repo, and the live site updates
itself automatically every time you push.

1. Create a new repo on GitHub (e.g. `thz-analyzer`) and push everything in
   this folder to it:
   ```
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/<your-username>/thz-analyzer.git
   git push -u origin main
   ```
2. On GitHub, go to **Settings → Pages** on the repo, and under "Build and
   deployment" set **Source** to **GitHub Actions**. That's it — the workflow
   in `.github/workflows/deploy.yml` (already included) will build and deploy
   automatically on every push to `main`.
3. After the first push, check the **Actions** tab for the workflow run. Once
   it finishes, your site is live at:
   ```
   https://<your-username>.github.io/thz-analyzer/
   ```
4. To update it later: edit `src/App.jsx` (or paste in an updated version from
   Claude), commit, and push. The site rebuilds and redeploys on its own —
   no manual build step needed.

**Note on paths:** this project builds to a single self-contained
`dist/index.html` (via `vite-plugin-singlefile`), so there are no separate
JS/CSS asset files to worry about breaking under GitHub Pages' subpath
(`/thz-analyzer/`) — the whole app is inlined into that one file.

### Alternative: Vercel or Netlify

Even less setup, and also auto-deploys on every push:

1. Push the repo to GitHub as above (skip the Pages step).
2. Go to [vercel.com](https://vercel.com) or [netlify.com](https://netlify.com),
   sign in with GitHub, and "Import" this repo.
3. Both auto-detect the Vite build (`npm run build`, output folder `dist`) —
   just accept the defaults and deploy.
4. You get a URL immediately (e.g. `thz-analyzer.vercel.app`), and it
   redeploys automatically on every push, including preview URLs for branches.

Either of these gives you exactly what you asked for: edit the source
whenever you like, push, and your labmates just open the same link — no
reinstall or re-share needed on their end.

## Option B — Single HTML file (no hosting, just a file to share)

If you'd rather not host it anywhere and just hand someone a file:

1. `npm install`
2. `npm run build`
3. Share the resulting `dist/index.html` directly (email, drive, USB). It's
   fully self-contained and works offline — double-click to open in a
   browser, no install needed.

## Option C — An actual desktop app / `.exe`

If you want something that feels like an installed program (taskbar icon, no
browser chrome), wrap the same build with **Tauri** (smaller, ~10–20 MB,
needs the Rust toolchain) or **Electron** (simpler setup, ~100+ MB output).
See the previous version of this README (or ask Claude) for step-by-step
commands — for a lab tool shared with a few people, Option A (hosted) or
Option B (single file) will get you there with far less to maintain.

## Notes

- `src/App.jsx` is the exact component from the Claude artifact — no changes
  needed. If you keep iterating on the tool with Claude, copy the updated
  file into `src/App.jsx`, commit, and push (Option A) or rebuild (Option B).
- If you add new libraries in Claude later (anything beyond React, Recharts,
  MathJS, PapaParse, and lucide-react), add them to `package.json`'s
  `dependencies` before rebuilding/pushing.
- No login or access control is set up — anyone with the link can open and
  use the hosted site. Fine for an internal lab tool passed around by URL;
  let me know if you actually need it private/password-gated and I can help
  set that up (e.g. Vercel/Netlify both support simple password protection
  on paid tiers, or a basic auth layer).
