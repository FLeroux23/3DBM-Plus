"""Module to manipulate geometry of pyvista meshes"""

import numpy as np
import pyvista as pv
from helpers.geometry import plane_params, project_mesh, to_3d
from scipy.spatial import distance_matrix
from sklearn.cluster import AgglomerativeClustering

def get_points_of_type(mesh, surface_type):
    """Returns the points that belong to the given surface type."""
    
    # Check if "semantics" exists in cell data
    semantics = mesh.cell_data.get("semantics")
    if semantics is None:
        return np.array([])

    # Create a boolean mask for the desired surface type
    idxs = np.array(semantics) == surface_type

    # Check if there are no matching indices
    if not np.any(idxs):
        return np.array([])

    # Use numpy array to collect points corresponding to the surface type
    points = np.array([mesh.get_cell(i).points for i in range(mesh.number_of_cells)], dtype=object)
    
    return np.vstack(points[idxs])

def move_to_origin(mesh):
    """Moves the object to the origin"""
    pts = mesh.points
    t = np.min(pts, axis=0)
    mesh.points -= t

    return mesh, t

def extrude(shape, min, max):
    """Create a pyvista mesh from a polygon"""

    points = np.array([[p[0], p[1], min] for p in shape.boundary.coords])
    mesh = pv.PolyData(points).delaunay_2d()

    if min == max:
        return mesh

    # Transform to 0, 0, 0 to avoid precision issues
    t = np.mean(mesh.points, axis=0)
    mesh.points -= t
    
    mesh = mesh.extrude([0.0, 0.0, max - min], capping=True)
    
    # Transform back to origina coords
    # mesh.points = mesh.points + t

    mesh = mesh.clean().triangulate()

    return mesh

def area_by_surface(mesh, sloped_angle_threshold=3, tri_mesh=None):
    """Compute the area per semantic surface"""

    sloped_threshold = np.cos(np.radians(sloped_angle_threshold))
    
    # Initialize dictionaries for area, point count, and surface count
    area = {
        "GroundSurface": 0,
        "WallSurface": 0,
        "RoofSurface": 0,
        "RoofSurfaceFlat": 0,
        "RoofSurfaceSloped": 0
    }

    point_count = {
        "GroundSurface": 0,
        "WallSurface": 0,
        "RoofSurface": 0
    }

    surface_count = {
        "GroundSurface": 0,
        "WallSurface": 0,
        "RoofSurface": 0,
        "RoofSurfaceFlat": 0,
        "RoofSurfaceSloped": 0
    }

    # Compute the triangulated surfaces to fix issues with areas
    if tri_mesh is None:
        tri_mesh = mesh.triangulate()

    if "semantics" in mesh.cell_data:
        # Compute area per surface type
        sized = tri_mesh.compute_cell_sizes()
        area_data = sized.cell_data["Area"]

        points_per_cell = np.array([mesh.get_cell(i).n_points for i in range(mesh.number_of_cells)])

        # Process each surface type
        for surface_type in ["GroundSurface", "WallSurface", "RoofSurface"]:
            triangle_idxs_mask = [s == surface_type for s in tri_mesh.cell_data["semantics"]]
            triangle_idxs = [i for i,s in enumerate(tri_mesh.cell_data["semantics"]) if s == surface_type]

            if surface_type == "RoofSurface":
                normals = sized.cell_normals[triangle_idxs]
                dot_products = normals.dot([0, 0, 1])
                sloped_mask = dot_products < sloped_threshold

                area["RoofSurfaceSloped"] += area_data[triangle_idxs][sloped_mask].sum()
                area["RoofSurfaceFlat"] += area_data[triangle_idxs][~sloped_mask].sum()
                area["RoofSurface"] += area_data[triangle_idxs].sum()
            else:
                area[surface_type] = area_data[triangle_idxs_mask].sum()
                
            face_idxs = [s == surface_type for s in mesh.cell_data["semantics"]]
            point_count[surface_type] = sum(points_per_cell[face_idxs])
            surface_count[surface_type] = sum(face_idxs)

    return area, point_count, surface_count

def face_planes(mesh):
    """Return the parameters (a, b, c, d) of all planes in a given mesh."""
    
    n_cells = mesh.n_cells
    face_normals = mesh.face_normals
    
    # Preallocate a NumPy array for plane parameters
    plane_params_list = np.empty((n_cells, 4))  # 4 parameters for each plane

    for i in range(n_cells):
        # Directly extract the first point of the cell
        cell_point = mesh.get_cell(i).points[0]
        plane_params_list[i] = plane_params(face_normals[i], cell_point)
    
    return plane_params_list

def cluster_meshes(meshes, threshold=0.1):
    """Clusters the faces of the given meshes."""
    
    n_meshes = len(meshes)
    
    # Compute the "absolute" plane params for every face of the meshes
    planes = [face_planes(mesh) for mesh in meshes]

    # Find the common planes between the faces
    all_planes = np.concatenate(planes)
    all_labels, n_clusters = cluster_faces(all_planes, threshold)
    
    # Precompute split indices for labels
    split_indices = np.cumsum([0] + [meshes[m].n_cells for m in range(n_meshes)])

    # Use split_indices to divide labels into the correct segments
    labels = [all_labels[split_indices[m]:split_indices[m + 1]] for m in range(n_meshes)]
    
    return labels, n_clusters

def cluster_faces(data, threshold=0.1):
    """Clusters the given planes"""
    ndata = np.array(data)
    
    dm1 = distance_matrix(ndata, ndata)
    dm2 = distance_matrix(ndata, -ndata)

    dist_mat = np.minimum(dm1, dm2)

    clustering = AgglomerativeClustering(n_clusters=None,
                                         distance_threshold=threshold,
                                         affinity='precomputed',
                                         linkage='average').fit(dist_mat)
    
    return clustering.labels_, clustering.n_clusters_

def cluster_faces_simple(data, threshold=0.1):
    """Clusters the given planes"""
    # we can delete the third column because it is all 0's for vertical planes
    ndata = np.delete(data, 2, 1)

    # flip normals so that they can not be pointing in opposite direction for same plane
    neg_x = ndata[:,0] < 0
    ndata[neg_x,:] = ndata[neg_x,:] * -1

    dist_mat = distance_matrix(ndata, ndata)

    clustering = AgglomerativeClustering(n_clusters=None,
                                         distance_threshold=threshold,
                                         metric='precomputed',
                                         linkage='average').fit(dist_mat)
    
    return clustering.labels_, clustering.n_clusters_

def cluster_faces_alternative(data, angle_threshold=0.005, dist_threshold=0.2):
    """Clusters the given planes"""
    def groupby(a, clusterids):
        return out, sidx

    ndata = np.array(data)

    # new method - angle clustering
    angle_clustering = AgglomerativeClustering(n_clusters=None,
                                         metric='cosine',
                                         distance_threshold=angle_threshold,
                                         linkage='average').fit(ndata[:,:3])
    # group angle clusters
    angle_clusters, remap = groupby(ndata[:,3:], angle_clustering.labels_)

    # get dist clusters for each angle cluster
    labels_ = np.empty(0, dtype=int)
    min_label = 0
    for angle_cluster in angle_clusters:
        if angle_cluster.size == 1:
            labels_ = np.hstack((labels_, min_label))
            min_label += 1
        else:
            dist_clustering = AgglomerativeClustering(n_clusters=None,
                                                metric='euclidean',
                                                distance_threshold=dist_threshold,
                                                linkage='average').fit(angle_cluster)
            labels_ = np.hstack((labels_, dist_clustering.labels_ + min_label))
            min_label = labels_.max()+1
    
    # re order back to input data order
    n_planes = ndata.shape[0]
    labels = np.empty(n_planes, dtype=int)
    labels[remap] = labels_

    n_clusters = (np.bincount(labels)!=0).sum()
    return labels, n_clusters

def intersect_surfaces(meshes):
    """Return the intersection between the surfaces of multiple meshes"""

    def get_area_from_ring(areas, area, geom, normal, origin, subtract=False):
        pts = to_3d(geom.coords, normal, origin)
        common_mesh = pv.PolyData(pts, faces=[len(pts)] + list(range(len(pts))))
        common_mesh["area"] = [-area] if subtract else [area]
        areas.append(common_mesh)

    def get_area_from_polygon(areas, geom, normal, origin):
        # polygon with holes:
        if geom.boundary.type == 'MultiLineString':
            get_area_from_ring(areas, geom.area, geom.boundary[0], normal, origin)
            for sgeom in geom.boundary[1:]:
                get_area_from_ring(areas, 0, sgeom, normal, origin, subtract=True)
        # polygon without holes:
        elif geom.boundary.type == 'LineString':
            get_area_from_ring(areas, geom.area, geom.boundary, normal, origin)
    
    n_meshes = len(meshes)
    
    areas = []
    
    labels, n_clusters = cluster_meshes(meshes)
    
    for plane in range(n_clusters):
        # For every common plane, extract the faces that belong to it
        idxs = [np.where(np.array(labels[m]) == plane)[0] for m in range(n_meshes)]

        # Check if any index list is empty; if so, continue to next plane
        if any(len(idx) == 0 for idx in idxs):
            continue
        
        msurfaces = [mesh.extract_cells(idxs[i]).extract_surface() for i, mesh in enumerate(meshes)]
                
        # Set the normal and origin point for a plane to project the faces
        origin = msurfaces[0].clean().points[0]
        normal = msurfaces[0].face_normals[0]
        
        # Create the two 2D polygons by projecting the faces
        polys = [project_mesh(msurface, normal, origin) for msurface in msurfaces]
        
        # Intersect the 2D polygons
        inter = polys[0]
        for poly in polys[1:]:
            inter = inter.intersection(poly)

        # Process intersection result
        if inter.area > 0.001:
            if inter.type in {"MultiPolygon", "GeometryCollection"}:
                for geom in inter.geoms:
                    if geom.type == "Polygon":
                        get_area_from_polygon(areas, geom, normal, origin)
            elif inter.type == "Polygon":
                get_area_from_polygon(areas, inter, normal, origin)
    
    return areas
