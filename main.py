# pip install -q emoji==2.* sentence-transformers umap-learn

import re, math, sys, json
import numpy as np

import emoji  # pypi "emoji"
from sentence_transformers import SentenceTransformer
import umap

# 1) collect the emoji set (all single codepoints and sequences known to the lib)
#    emoji.EMOJI_DATA keys are the grapheme clusters we want.
all_emoji = list(emoji.EMOJI_DATA.keys())

# Build a canonical set: default (yellow) skin tone and gender-neutral if available,
# otherwise the man version. Group similar emojis to avoid redundancies.
tone_pat = re.compile(r"_(light|medium_light|medium|medium_dark|dark)_skin_tone")

def alias_of(e: str) -> str | None:
    a = emoji.demojize(e)
    if not (a.startswith(":") and a.endswith(":")):
        return None
    return a.strip(":")

def strip_tone(alias: str) -> str:
    return tone_pat.sub("", alias)

def group_key(alias: str) -> str:
    toks = alias.split("_")
    mapped = [
        ("person" if t in {"man", "men", "boy", "woman", "women", "girl"} else t)
        for t in toks
    ]
    return "_".join(mapped)

from collections import OrderedDict
groups: "OrderedDict[str, dict]" = OrderedDict()

for e in all_emoji:
    a = alias_of(e)
    if not a:
        continue
    # Skip keycaps/variation selectors and other noisy categories
    if re.search(r"keycap|variation selector", a, re.I):
        continue
    a0 = strip_tone(a)
    gk = group_key(a0)
    entry = groups.get(gk)
    if entry is None:
        entry = {"person": None, "man": None, "any": None}
        groups[gk] = entry
    tokens = set(a0.split("_"))
    if "person" in tokens and entry["person"] is None:
        entry["person"] = {"alias": a0, "e": e}
    elif "man" in tokens and entry["man"] is None:
        entry["man"] = {"alias": a0, "e": e}
    elif entry["any"] is None:
        entry["any"] = {"alias": a0, "e": e}

# Select preferred representative per group
selected = []
for gk, entry in groups.items():
    choice = entry["person"] or entry["man"] or entry["any"]
    if not choice:
        continue
    # Try to emojize the chosen alias without skin tone to ensure default/yellow.
    try:
        e = emoji.emojize(f":{choice['alias']}:")
        if e.startswith(":") and e.endswith(":"):
            # emojize failed, fall back to original grapheme
            e = choice["e"]
    except Exception:
        e = choice["e"]
    selected.append((choice["alias"], e))

emjs = [e for _, e in selected]

# 2) get semantic strings to embed. demojize gives :face_with_tears_of_joy: → we normalize.
def label_for(e: str) -> str:
    alias = emoji.demojize(e).strip(":")
    # nicer text for the model
    return alias.replace("_", " ").replace("-", " ")

texts = [label_for(e) for e in emjs]

# 3) embed
# any decent general text-embedder works; picking a tiny, fast one
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
emb = model.encode(texts, normalize_embeddings=True, batch_size=256, show_progress_bar=True)

# 4) project to 2d with umap (tsne also works; umap is faster & keeps local structure)
proj = umap.UMAP(
    n_neighbors=30,
    min_dist=0.25,
    metric="cosine",
    random_state=42,
).fit_transform(emb)

# 5) write an interactive HTML using Plotly (no Python plotting libs required)
outfile = "emoji_map.html"

# Prepare data as JSON for embedding in the HTML
data = {
    "x": proj[:, 0].astype(float).tolist(),
    "y": proj[:, 1].astype(float).tolist(),
    "emoji": emjs,
    "label": texts,
}

# Compute axis ranges with a small padding
xmin, xmax = float(np.min(proj[:, 0])), float(np.max(proj[:, 0]))
ymin, ymax = float(np.min(proj[:, 1])), float(np.max(proj[:, 1]))
xpad = (xmax - xmin) * 0.05 or 0.5
ypad = (ymax - ymin) * 0.05 or 0.5

html = f"""
<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>emoji semantic map</title>
  <style>
    html, body {{ height: 100%; margin: 0; }}
    #plot {{ position: relative; width: 100%; height: 100%; background: #fff; }}
    .tooltip {{
      position: absolute; pointer-events: none; display: none;
      background: rgba(0,0,0,0.8); color: #fff; padding: 4px 6px;
      border-radius: 4px; font: 12px/1.4 -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
      white-space: nowrap;
    }}
  </style>
  <script src=\"https://cdn.jsdelivr.net/npm/d3@7\"></script>
  <script src=\"https://twemoji.maxcdn.com/v/latest/twemoji.min.js\" crossorigin=\"anonymous\"></script>
</head>
<body>
  <div id=\"plot\"></div>
  <script>
    const data = {json.dumps(data)};
    const xs = data.x, ys = data.y, emjs = data.emoji, labels = data.label;
    let curXs = xs.slice(), curYs = ys.slice();
    const n = xs.length;
    const padding = 20;
    const baseSize = 18;
    const dpr = window.devicePixelRatio || 1;
    const xmin = {xmin}, xmax = {xmax};
    const ymin = {ymin}, ymax = {ymax};

    const container = document.getElementById('plot');
    const canvas = document.createElement('canvas');
    container.appendChild(canvas);
    const ctx = canvas.getContext('2d');

    const tooltip = document.createElement('div');
    tooltip.className = 'tooltip';
    container.appendChild(tooltip);

    let transform = d3.zoomIdentity;
    const xScale = d3.scaleLinear().domain([xmin, xmax]);
    const yScale = d3.scaleLinear().domain([ymin, ymax]);

    function resize() {{
      const rect = container.getBoundingClientRect();
      canvas.style.width = rect.width + 'px';
      canvas.style.height = rect.height + 'px';
      canvas.width = Math.max(1, Math.round(rect.width * dpr));
      canvas.height = Math.max(1, Math.round(rect.height * dpr));
      draw();
    }}

    function dataToPixel(x, y) {{
      const px = transform.applyX(xScale(x));
      const py = transform.applyY(yScale(y));
      return [px, py];
    }}

    const imgCache = new Map();
    function urlForEmoji(e) {{
      const code = twemoji.convert.toCodePoint(e);
      // 72x72 PNGs render consistently on canvas
      return 'https://twemoji.maxcdn.com/v/latest/72x72/' + code + '.png';
    }}

    function loadImages() {{
      const unique = Array.from(new Set(emjs));
      return Promise.all(unique.map(e => new Promise((resolve) => {{
        if (imgCache.has(e)) return resolve();
        const img = new Image();
        img.crossOrigin = 'anonymous';
        img.onload = () => {{ imgCache.set(e, img); resolve(); }};
        img.onerror = () => {{ imgCache.set(e, null); resolve(); }};
        img.src = urlForEmoji(e);
      }})));
    }}

    function draw() {{
      const rect = canvas.getBoundingClientRect();
      xScale.range([padding, rect.width - padding]);
      yScale.range([rect.height - padding, padding]);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, rect.width, rect.height);
      for (let i = 0; i < n; i++) {{
        const img = imgCache.get(emjs[i]);
        const [px, py] = dataToPixel(curXs[i], curYs[i]);
        const size = baseSize * transform.k;
        if (img) {{
          ctx.drawImage(img, px - size/2, py - size/2, size, size);
        }} else {{
          ctx.fillStyle = '#666';
          ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
          ctx.font = (size) + 'px sans-serif';
          ctx.fillText('?', px, py);
        }}
      }}
    }}

    // One-time layout relaxation to reduce overlaps in screen space
    let relaxed = false;
    function relax() {{
      const rect = canvas.getBoundingClientRect();
      xScale.range([padding, rect.width - padding]);
      yScale.range([rect.height - padding, padding]);
      const nodes = new Array(n);
      for (let i = 0; i < n; i++) {{
        nodes[i] = {{ i, x: xScale(xs[i]), y: yScale(ys[i]) }};
      }}
      const collideR = baseSize * 0.7;
      const anchor = 0.2; // how strongly to stay near original
      const sim = d3.forceSimulation(nodes)
        .force('x', d3.forceX(d => nodes[d.i].x).strength(anchor))
        .force('y', d3.forceY(d => nodes[d.i].y).strength(anchor))
        .force('collide', d3.forceCollide(collideR).iterations(2))
        .stop();
      for (let t = 0; t < 200; t++) sim.tick();
      // Map back to data space
      curXs = new Array(n); curYs = new Array(n);
      for (let i = 0; i < n; i++) {{
        curXs[i] = xScale.invert(nodes[i].x);
        curYs[i] = yScale.invert(nodes[i].y);
      }}
      relaxed = true;
      draw();
    }}

    const zoom = d3.zoom().scaleExtent([0.5, 20]).on('zoom', (ev) => {{
      transform = ev.transform;
      draw();
    }});
    d3.select(canvas).call(zoom);

    canvas.addEventListener('mousemove', (ev) => {{
      const rect = canvas.getBoundingClientRect();
      const mx = (ev.clientX - rect.left);
      const my = (ev.clientY - rect.top);
      let best = -1, bestD2 = 400; // px^2
      for (let i = 0; i < n; i++) {{
        const [px, py] = dataToPixel(xs[i], ys[i]);
        const dx = px - mx, dy = py - my;
        const d2 = dx*dx + dy*dy;
        if (d2 < bestD2) {{ bestD2 = d2; best = i; }}
      }}
      if (best >= 0) {{
        tooltip.textContent = labels[best];
        tooltip.style.display = 'block';
        tooltip.style.left = (mx + 12) + 'px';
        tooltip.style.top = (my + 12) + 'px';
      }} else {{
        tooltip.style.display = 'none';
      }}
    }});
    canvas.addEventListener('mouseleave', () => {{ tooltip.style.display = 'none'; }});

    window.addEventListener('resize', () => {{
      // On resize, reset to base data positions and redraw
      curXs = xs.slice();
      curYs = ys.slice();
      relaxed = false;
      resize();
    }});
    loadImages().then(() => {{ resize(); relax(); }});
  </script>
  </body>
  </html>
"""

with open(outfile, "w", encoding="utf-8") as f:
    f.write(html)

print(f"wrote {outfile}")
