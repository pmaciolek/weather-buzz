"""Bundle web/index.html + its data, scripts, base map and fonts into one HTML file
that works offline: web/snapshots/cycling-weather-atlas-<date>.html

Usage: python3.13 snapshot.py
Third-party assets are downloaded once into cache/vendor/.
"""
import base64
import datetime as dt
import json
import re
import urllib.request

import config as C

VENDOR = C.CACHE / "vendor"
SCRIPTS = {
    "https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js": "d3-7.9.0.min.js",
    "https://cdn.jsdelivr.net/npm/topojson-client@3.1.0/dist/topojson-client.min.js": "topojson-client-3.1.0.min.js",
}
WORLD_URL = "https://cdn.jsdelivr.net/npm/world-atlas@2.0.2/countries-50m.json"
# A modern browser UA makes Google Fonts serve woff2.
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
# Polish, Spanish, Portuguese, Croatian names need latin-ext as well as latin.
FONT_SUBSETS = ("latin", "latin-ext")


def _get(url, name=None, ua=False):
    path = VENDOR / (name or re.sub(r"[^A-Za-z0-9._-]", "_", url)[-120:])
    if not path.exists():
        VENDOR.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(url, headers={"User-Agent": UA} if ua else {})
        with urllib.request.urlopen(req, timeout=60) as r:
            path.write_bytes(r.read())
    return path.read_bytes()


def inline_fonts(css_url):
    css = _get(css_url, "fonts.css", ua=True).decode()
    out = []
    # Google's CSS is a sequence of "/* subset */ @font-face {...}" blocks.
    for subset, block in re.findall(r"/\* ([\w-]+) \*/\s*(@font-face\s*{[^}]*})", css):
        if subset not in FONT_SUBSETS:
            continue
        url = re.search(r"url\((https://[^)]+\.woff2)\)", block).group(1)
        data = base64.b64encode(_get(url)).decode()
        out.append(block.replace(url, f"data:font/woff2;base64,{data}"))
    return "\n".join(out)


def main():
    html = (C.ROOT / "web" / "index.html").read_text()
    html = re.sub(r'<link rel="preconnect"[^>]*>\n?', "", html)
    css_url = re.search(r'<link rel="stylesheet" href="(https://fonts\.googleapis\.com/[^"]+)">', html).group(1)
    html = html.replace(f'<link rel="stylesheet" href="{css_url}">', f"<style>\n{inline_fonts(css_url.replace('&amp;', '&'))}\n</style>")
    for url, name in SCRIPTS.items():
        js = _get(url, name).decode()
        html = html.replace(f'<script src="{url}"></script>', f"<script>{js}</script>")
    assert "cdn.jsdelivr.net/npm/d3" not in html.split("<script>// A self")[0].split("</head>")[0], "script not inlined"

    grid_path = C.WEB_DATA / "grid.json"
    snap = {
        "cities": json.loads((C.WEB_DATA / "cities.json").read_text()),
        "grid": json.loads(grid_path.read_text()) if grid_path.exists() else None,
        "world": json.loads(_get(WORLD_URL, "countries-50m.json")),
    }
    # "</" must not appear inside an inline script.
    payload = json.dumps(snap, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    html = html.replace("<body>", f"<body>\n<script>window.SNAPSHOT = {payload};</script>", 1)

    out_dir = C.ROOT / "web" / "snapshots"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"cycling-weather-atlas-{dt.datetime.now():%Y-%m-%d-%H%M}.html"
    out.write_text(html)
    left = [u for u in re.findall(r'(?:src|href)="(https?://[^"]+)"', html) if "fonts" in u or u.endswith(".js")]
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB); external assets left: {left or 'none'}")


if __name__ == "__main__":
    main()
