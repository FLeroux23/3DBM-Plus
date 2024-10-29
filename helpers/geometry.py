"""Module with functions for 3D geometrical operations"""

import numpy as np
import mapbox_earcut as earcut
import pyvista as pv
from shapely.geometry import Polygon, MultiPolygon

def surface_normal(poly):
    """Calculate the surface normal of a polygon defined by its vertices."""
    
    n = np.zeros(3)
    
    # Iterate through vertices to compute the normal
    num_vertices = len(poly)
    for i in range(num_vertices):
        v_curr = poly[i]
        v_next = poly[(i + 1) % num_vertices]

        n[0] += (v_curr[1] - v_next[1]) * (v_curr[2] + v_next[2])
        n[1] += (v_curr[2] - v_next[2]) * (v_curr[0] + v_next[0])
        n[2] += (v_curr[0] - v_next[0]) * (v_curr[1] + v_next[1])

    # Check for zero-length normal
    norm_length = np.linalg.norm(n)
    if norm_length == 0:  # Prevent division by zero
        return -1

    normalized = n / norm_length

    return (n / np.linalg.norm(n)).tolist()

def compute_surface_area(mesh):
    def triangle_area(v1, v2, v3):
        # Calculate the area of a triangle given its vertices
        a = np.linalg.norm(v2 - v1)
        b = np.linalg.norm(v3 - v1)
        c = np.linalg.norm(v3 - v2)
        s = (a + b + c) / 2.0  # Semi-perimeter
        area_term = s * (s - a) * (s - b) * (s - c)  # Heron's formula
        
        # Check for negative area term due to floating-point precision issues
        if area_term < 0:
            return 0
    
        area = np.sqrt(area_term)
        
        return area

    boundaries = mesh.faces
    vertices = mesh.points

    areas = []

    # Use an index to track the current position in the faces array
    index = 0
    while index < boundaries.size:
        num_vertices = boundaries[index]  # Get the number of vertices for the current face
        index += 1  # Move to the first vertex index
        face_indices = boundaries[index:index + num_vertices]  # Get the vertex indices
        points = vertices[face_indices]  # Get the vertex coordinates
        
        # Calculate area by triangulating the polygon
        area = 0.0
        for i in range(1, num_vertices - 1):
            area += triangle_area(points[0], points[i], points[i + 1])
        
        areas.append(area)  # Append the area to the list
        index += num_vertices  # Move to the next face

    return np.array(areas)

def axes_of_normal(normal):
    """Returns an x-axis and y-axis on a plane of the given normal"""
    
    normal = np.array(normal)  # Ensure normal is a NumPy array for vector operations

    if abs(normal[2]) > 0.001:  # Check z-component
        x_axis = np.array([1, 0, -normal[0] / normal[2]])
    elif abs(normal[1]) > 0.001:  # Check y-component
        x_axis = np.array([1, -normal[0] / normal[1], 0])
    else:  # Default case
        x_axis = np.array([-normal[1] / normal[0], 1, 0])
    
    x_axis /= np.linalg.norm(x_axis)  # Normalize x_axis
    y_axis = np.cross(normal, x_axis)  # Compute y_axis as cross product

    return x_axis, y_axis

def project_2d(points, normal, origin=None):
    """Project 3D points onto a 2D plane defined by the given normal vector"""
    
    if origin is None:
        origin = points[0]

    x_axis, y_axis = axes_of_normal(normal)
    
    # Convert points to a NumPy array for vectorized operations
    points_array = np.array(points)

    # Center points by the origin
    centered_points = points_array - origin

    # Compute 2D projections
    projected = np.column_stack((
        np.dot(centered_points, x_axis),
        np.dot(centered_points, y_axis)
    ))
    
    return projected.tolist()

def triangulate(mesh):
    """Triangulates a mesh in the proper way"""
    
    final_mesh = pv.PolyData()
    n_cells = mesh.n_cells

    # Collect cells to be triangulated
    for i in range(n_cells):
        if not mesh.get_cell(i).type in {5, 6, 7, 9, 10}:
            continue

        pts = mesh.get_cell(i).points
        p = project_2d(pts, mesh.face_normals[i])
        result = earcut.triangulate_float32(p, [len(p)])

        # Reshape and prepare triangles
        triangles = result.reshape(-1, 3)
        t_count = triangles.shape[0]
        triangles_flat = np.hstack([[3] + list(t) for t in triangles])
        
        # Create new mesh for the triangles
        new_mesh = pv.PolyData(pts, triangles_flat)
        
        # Efficiently copy cell data
        for k in mesh.cell_data:
            new_mesh[k] = np.tile(mesh.cell_data[k][i], (t_count, 1))

        # Concatenate the new mesh to the final mesh
        final_mesh += new_mesh
    
    return final_mesh

def triangulate_polygon(face, vertices, offset = 0):
    """Returns the points and triangles for a given CityJSON polygon"""

    points = vertices[np.hstack(face)]
    normal = surface_normal(points)
    holes = [0]
    for ring in face:
        holes.append(len(ring) + holes[-1])
    holes = holes[1:]

    points_2d = project_2d(points, normal)

    result = earcut.triangulate_float32(points_2d, holes)

    result += offset

    t_count = len(result.reshape(-1,3))
    if t_count == 0:
        return points,  []
    triangles = np.hstack([[3] + list(t) for t in result.reshape(-1,3)])

    return points, triangles

def plane_params(normal, origin, rounding=2):
    """Returns the params (a, b, c, d) of the plane equation for the given
    normal and origin point.
    """
    a, b, c = np.round_(normal, 3)
    x0, y0, z0 = origin

    d = -(a * x0 + b * y0 + c * z0)

    d = round(d, rounding) if rounding >= 0 else d

    return np.array([a, b, c, d])

def project_mesh(mesh, normal, origin):
    """Project the faces of a mesh to the given plane"""

    p = [None] * mesh.n_cells

    for i in range(mesh.n_cells):
        pts = mesh.get_cell(i).points
        pts_2d = project_2d(pts, normal, origin)
        
        p[i] = Polygon(pts_2d)

    return MultiPolygon(p).buffer(0)

def to_3d(points, normal, origin):
    """Returns the 3d coordinates of a 2d points from a given plane"""

    xa, ya = axes_of_normal(normal)
    
    mat = np.array([xa, ya])
    pts = np.array(points)
    
    return np.dot(pts, mat) + origin
