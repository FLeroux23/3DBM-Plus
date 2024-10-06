import numpy as np
import trimesh as tm
import pyvista as pv

def to_trimesh(mesh):
    """Returns a trimesh from a pyvista PolyData"""
    v = mesh.points
    f = mesh.faces.reshape(-1, 4)[:, 1:]

    return tm.Trimesh(v, f)

def to_pyvista(mesh):
    """Return a PolyData from a trimesh"""
    v = mesh.vertices
    f = mesh.faces
    
    f = np.hstack([[len(f)] + list(f) for f in mesh.faces])
    
    return pv.PolyData(v, f, len(mesh.faces))

def intersect(mesh1, mesh2, engine="manifold"):
    """Returns the intersection of two meshes (in trimesh format)"""

    return tm.boolean.intersection([mesh1, mesh2], engine=engine)
    
def union(mesh1, mesh2, engine="manifold"):
    """Returns the union of two meshes (in trimesh format)"""

    return tm.boolean.union([mesh1, mesh2], engine=engine)

def difference(mesh1, mesh2, engine="manifold"):
    """Returns the difference between two volumes (in trimesh format)"""

    return tm.boolean.difference([mesh1, mesh2], engine=engine)

def symmetric_difference(mesh1, mesh2, engine="manifold"):
    """Returns the symmetric difference of two volumes (in trimesh format)"""

    u = union(mesh1, mesh2, engine)
    i = intersect(mesh1, mesh2, engine)

    return difference(u, i, engine)
