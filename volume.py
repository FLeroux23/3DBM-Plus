import numpy as np
import pyvista as pv
import scipy.spatial as ss
import sys

import json

filename = sys.argv[1]

with open(filename, 'r') as f:
    cm = json.load(f)

if "transform" in cm:
    s = cm["transform"]["scale"]
    t = cm["transform"]["translate"]
    verts = [[v[0] * s[0] + t[0], v[1] * s[1] + t[1], v[2] * s[2] + t[2]] for v in cm["vertices"]]
else:
    verts = cm["vertices"]

# mesh points
vertices = np.array(verts)

print("id, actual volume, convex hull volume, area")
for obj in cm["CityObjects"]:
    building = cm["CityObjects"][obj]

    if not "geometry" in building or len(building["geometry"]) == 0:
        continue

    geom = building["geometry"][0]

    if geom["type"] == "MultiSurface":
        boundaries = geom["boundaries"]
    elif geom["type"] == "Solid":
        boundaries = geom["boundaries"][0]
    else:
        continue

    f = [[len(r[0])] + r[0] for r in [f for f in boundaries]]
    # mesh faces
    faces = np.hstack(f)    # triangle

    surf = pv.PolyData(vertices, faces)

    ch_volume = ss.ConvexHull([verts[i] for i in np.array(geom["boundaries"]).flatten()]).volume

    print(f"{obj}, {surf.volume}, {ch_volume}, {surf.area}")