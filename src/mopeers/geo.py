"""Границы действующих МО для карты лендинга: gpkg СберИндекса → упрощённый TopoJSON.

Упрощение — mapshaper (npx): сохраняет общие границы соседних МО, 4% вершин, 1,3 МБ на 2 594 МО.
"""
import subprocess
import tempfile
from pathlib import Path

import geopandas as gpd

from .config import load_config, path


def build(cfg=None):
    cfg = cfg or load_config()
    g = gpd.read_file(path(cfg, "mo_poly"))
    g = g[g.year_to == 9999][["territory_id", "geometry"]]
    out = path(cfg, "outputs") / "geo"
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "mo.geojson"
        g.to_file(src, driver="GeoJSON")
        subprocess.run(f'npx -y mapshaper@0.6 "{src}" -simplify {cfg["geo"]["simplify"]} weighted keep-shapes -clean '
                       f'-o "{out / "mo.topojson"}" format=topojson quantization=1e5', shell=True, check=True)
    print("МО на карте", len(g))
