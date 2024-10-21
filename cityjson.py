"""Module with functions for manipulating CityJSON data"""

import numpy as np
from shapely.geometry import Polygon, MultiPolygon
from helpers.geometry import project_2d, surface_normal, triangulate, triangulate_polygon
import pyvista as pv

def get_surface_boundaries(geom):
    """Returns the boundaries for all surfaces"""

    geom_type = geom["type"]
    
    if geom_type in {"MultiSurface", "CompositeSurface"}:
        return geom["boundaries"]
    elif geom_type == "Solid":
        return geom["boundaries"][0]
    
    raise ValueError("Geometry not supported")

def get_points(geom, verts):
    """Return the points of the geometry."""
    
    boundaries = get_surface_boundaries(geom)
    
    # Flatten the list of indices and map them to points in 'verts'
    points = [verts[v] for ring in boundaries for v in ring[0]]
    
    return points

def to_shapely(geom, vertices, ground_only=True):
    """Returns a shapely geometry of the footprint from a CityJSON geometry"""

    boundaries = get_surface_boundaries(geom)

    if ground_only and "semantics" in geom:
        semantics = geom["semantics"]
        values = semantics["values"]
        if geom["type"] != "MultiSurface":
            values = values[0]
        
        ground_idxs = [semantics["surfaces"][i]["type"] == "GroundSurface" for i in values]
        boundaries = np.array(boundaries, dtype=object)[ground_idxs]
    
    shape = MultiPolygon([Polygon([vertices[v] for v in boundary[0]]) for boundary in boundaries])
    
    return shape.buffer(0)

def to_polydata(geom, vertices):
    """Returns the polydata mesh from a CityJSON geometry"""

    boundaries = get_surface_boundaries(geom)

    # Construct faces by prepending the count of vertices for each boundary
    faces = np.hstack([[len(r[0])] + r[0] for r in boundaries])

    mesh = pv.PolyData(vertices, faces)

    if "semantics" in geom:        
        semantics = geom["semantics"]
        values = semantics["values"]
        if geom["type"] != "MultiSurface":
            values = values[0]
        
        mesh.cell_data["semantics"] = [semantics["surfaces"][i]["type"] for i in values]
    
    return mesh

def to_triangulated_polydata(geom, vertices, clean=True):
    """Returns the polydata mesh from a CityJSON geometry"""

    boundaries = get_surface_boundaries(geom)
    
    if "semantics" in geom:        
        semantics = geom["semantics"]
        values = semantics["values"]
        if geom["type"] != "MultiSurface":
            values = values[0]
        semantic_types = [semantics["surfaces"][i]["type"] for i in values]

    # Initialize points, triangles, and semantics lists
    points = []
    triangles = []
    semantics = []

    triangle_count = 0
    
    # Process each boundary to triangulate
    for fid, face in enumerate(boundaries):
        try:
            new_points, new_triangles = triangulate_polygon(face, vertices, len(points))
        except:
            continue

        points.extend(new_points)
        triangles.extend(new_triangles)

        t_count = len(new_triangles) // 4  # Integer division to get triangle count
        triangle_count += t_count

        if "semantics" in geom:
            semantics.extend([semantic_types[fid]] * t_count)
    
    mesh = pv.PolyData(points, triangles)
    
    # Assign semantics to the mesh if available
    if "semantics" in geom:
        mesh["semantics"] = semantics
    
    # Clean the mesh if required
    if clean:
        mesh = mesh.clean()

    return mesh

def get_bbox(geom, verts):
    """Returns the bounding box of the geometry"""

    pts = np.array(get_points(geom, verts))
    
    min_vals = np.min(pts, axis=0)
    max_vals = np.max(pts, axis=0)

    # Combine min and max into a list of lists
    bbox = np.concatenate((min_vals[:, np.newaxis], max_vals[:, np.newaxis]), axis=1).flatten()

    return bbox
