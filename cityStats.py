# -*- coding: utf-8 -*-
import ast
import json
import math
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy

import click
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvista as pv
import rtree.index
import scipy.spatial as ss
from pymeshfix import MeshFix
from scipy import stats
from tqdm import tqdm
# 3dbm functions
import cityjson
import geometry
import shape_index as si
from helpers.geometry import surface_normal, compute_surface_area

# ---
# Functions
def compute_surface_normal(boundaries, vertices):
    normals = np.zeros((len(boundaries), 3))
    
    for fid, face in enumerate(boundaries):
        points = vertices[np.hstack(face)]
        normal = surface_normal(points)
        normals[fid] = normal

    return normals

def filter_by_semantic_surface(dataset, surface_data, semantic_type):
    semantics = np.array(dataset.cell_data["semantics"])
    filter_idxs = (semantics == semantic_type)

    surface_data = surface_data[filter_idxs]
    
    return surface_data, filter_idxs

def filter_level_of_detail(cm, lod):
    for cityobject_id in cm["CityObjects"]:
        cityobject = cm["CityObjects"][cityobject_id]

        new_geom = []

        for geom in cityobject["geometry"]:
            if str(geom["lod"]) == str(lod):
                new_geom.append(geom)

        cityobject["geometry"] = new_geom

def filter_building_type(cm, type='BuildingPart'):
    filtered_objects = {
        key: value for key, value in cm["CityObjects"].items() if value["type"] == type
    }
    
    cm["CityObjects"] = filtered_objects

def get_bearings(values, num_bins, weights):
    """Divides the values depending on the bins"""

    n = num_bins * 2

    bins = np.arange(n + 1) * 360 / n

    count, bin_edges = np.histogram(values, bins=bins, weights=weights)

    # move last bin to front, so eg 0.01° and 359.99° will be binned together
    count = np.roll(count, 1)
    bin_counts = count[::2] + count[1::2]

    # because we merged the bins, their edges are now only every other one
    bin_edges = bin_edges[range(0, len(bin_edges), 2)]

    return bin_counts, bin_edges

def get_wall_bearings(dataset, num_bins):
    """Returns the bearings of the azimuth angle of the normals for vertical
    surfaces of a dataset"""

    normals = dataset.face_normals

    if "semantics" in dataset.cell_data:
        wall_idxs = [s == "WallSurface" for s in dataset.cell_data["semantics"]]
    else:
        wall_idxs = [n[2] == 0 for n in normals]

    normals = normals[wall_idxs]

    azimuth = [get_azimuth(n[0], n[1]) for n in normals]

    sized = dataset.compute_cell_sizes()
    surface_areas = sized.cell_data["Area"][wall_idxs]

    return get_bearings(azimuth, num_bins, surface_areas)

def get_roof_bearings(dataset, num_bins):
    """Returns the bearings of the (vertical surfaces) of a dataset"""

    normals = dataset.face_normals

    if "semantics" in dataset.cell_data:
        roof_idxs = [s == "RoofSurface" for s in dataset.cell_data["semantics"]]
    else:
        roof_idxs = [n[2] > 0 for n in normals]

    normals = normals[roof_idxs]

    xz_angle = [get_azimuth(n[0], n[2]) for n in normals]
    yz_angle = [get_azimuth(n[1], n[2]) for n in normals]

    sized = dataset.compute_cell_sizes()
    surface_areas = sized.cell_data["Area"][roof_idxs]

    xz_counts, bin_edges = get_bearings(xz_angle, num_bins, surface_areas)
    yz_counts, bin_edges = get_bearings(yz_angle, num_bins, surface_areas)

    return xz_counts, yz_counts, bin_edges

def orientation_plot(
    bin_counts,
    bin_edges,
    num_bins=36,
    title=None,
    title_y=1.05,
    title_font=None,
    show=False
):
    if title_font is None:
        title_font = {"family": "DejaVu Sans", "size": 12, "weight": "bold"}

    width = 2 * np.pi / num_bins

    positions = np.radians(bin_edges[:-1])

    radius = bin_counts / bin_counts.sum()

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw={"projection": "polar"})
    ax.set_theta_zero_location("N")
    ax.set_theta_direction("clockwise")
    ax.set_ylim(top=radius.max())

    # configure the y-ticks and remove their labels
    ax.set_yticks(np.linspace(0, radius.max(), 5))
    ax.set_yticklabels(labels="")

    # configure the x-ticks and their labels
    xticklabels = ["N", "", "E", "", "S", "", "W", ""]
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(labels=xticklabels)
    ax.tick_params(axis="x", which="major", pad=-2)

    # draw the bars
    ax.bar(
        positions,
        height=radius,
        width=width,
        align="center",
        bottom=0,
        zorder=2
    )

    if title:
        ax.set_title(title, y=title_y, fontdict=title_font)

    if show:
        plt.show()
    
    return plt

def get_surface_plot(
    dataset,
    num_bins=36,
    title=None,
    title_y=1.05,
    title_font=None
):
    """Returns a plot for the surface normals of a polyData"""
    
    bin_counts, bin_edges = get_wall_bearings(dataset, num_bins)

    return orientation_plot(bin_counts, bin_edges)

def get_azimuth(dx, dy):
    """Returns the azimuth angle for the given coordinates"""
    
    return np.degrees(np.arctan2(dx, dy)) % 360

def get_elevation_angle(dx, dy, dz):
    """Returns the inclination angle for the given coordinates."""

    magnitude = np.sqrt(dx**2 + dy**2 + dz**2)

    angle_rad = np.arccos(dz / magnitude)

    angle_deg = np.degrees(angle_rad)

    return angle_deg

def compute_stats(values, percentile = 90, percentage = 75):
    """
    Returns the stats (mean, median, max, min, range etc.) for a set of values.
    """
    mean = np.mean(values)
    median = np.median(values)
    max_val = np.max(values)
    min_val = np.min(values)
    range_val = max_val - min_val
    std_dev = np.std(values)
    
    percentile = np.percentile(values, percentile)
    percentage = (percentage/100.0) * range_val + min_val
    
    mode_result = stats.mode(values, keepdims=False)
    mode_values = mode_result.mode
    mode_count = mode_result.count
    
    hDic = {
        'Mean': mean,
        'Median': median,
        'Max': max_val,
        'Min': min_val,
        'Range': range_val,
        'Std': std_dev,
        'Percentile': percentile,
        'Percentage': percentage,
        'Mode': mode_values if mode_count > 1 else mean,
        'ModeStatus': 'Y' if mode_count > 1 else 'N'

    }
        
    return hDic

def get_parent_attributes(cm):
    building_attributes = {
        obj_id: obj["attributes"]
        for obj_id, obj in cm["CityObjects"].items() if obj["type"] == "Building"
    }
    
    return building_attributes

def convexhull_volume(points):
    """Returns the volume of the convex hull"""

    try:
        return ss.ConvexHull(points).volume
    except:
        return 0

def boundingbox_volume(points):
    """Returns the volume of the bounding box"""
    
    minx = min(p[0] for p in points)
    maxx = max(p[0] for p in points)
    miny = min(p[1] for p in points)
    maxy = max(p[1] for p in points)
    minz = min(p[2] for p in points)
    maxz = max(p[2] for p in points)

    return (maxx - minx) * (maxy - miny) * (maxz - minz)

def get_errors_from_report(report, cityobject_id, cm):
    """Return the report for the feature of the given obj"""

    if not "features" in report:
        return []
    
    fid = cityobject_id

    cityobject = cm["CityObjects"][cityobject_id]
    primidx = 0

    if not "geometry" in cityobject or len(cityobject["geometry"]) == 0:
        return []

    if "parents" in cityobject:
        parid = cityobject["parents"][0]

        primidx = cm["CityObjects"][parid]["children"].index(cityobject_id)
        fid = parid

    for f in report["features"]:
        if f["id"] == fid:
            if "errors" in f["primitives"][primidx]:
                return list(map(lambda e: e["code"], f["primitives"][primidx]["errors"]))
            else:
                return []

    return []

def validate_report(report, cm):
    """Returns true if the report is actually for this file"""

    # TODO: Actually validate the report and that it corresponds to this cm
    return True

def tree_generator_function(cm, verts):
    for i, objid in enumerate(cm["CityObjects"]):
        obj = cm["CityObjects"][objid]

        if len(obj["geometry"]) == 0:
            continue

        xmin, xmax, ymin, ymax, zmin, zmax = cityjson.get_bbox(obj["geometry"][0], verts)
        yield (i, (xmin, ymin, zmin, xmax, ymax, zmax), objid)

def get_neighbours(cm, building_id, r, verts):
    """Return the neighbours of the given building"""

    building = cm["CityObjects"][building_id]

    if len(building["geometry"]) == 0:
        return []
    
    geom = building["geometry"][0]
    xmin, xmax, ymin, ymax, zmin, zmax = cityjson.get_bbox(geom, verts)
    objids = [n.object
            for n in r.intersection((xmin,
                                    ymin,
                                    zmin,
                                    xmax,
                                    ymax,
                                    zmax),
                                    objects=True)
            if n.object != building_id]

    if len(objids) == 0:
        objids = [n.object for n in r.nearest((xmin, ymin, zmin, xmax, ymax, zmax), 5, objects=True) if n.object != building_id]

    return [cm["CityObjects"][objid]["geometry"][0] for objid in objids]

class StatValuesBuilder:

    def __init__(self, values, indices_list) -> None:
        self.__values = values
        self.__indices_list = indices_list

    def compute_index(self, index_name):
        """Returns True if the given index is supposed to be computed"""

        return self.__indices_list is None or index_name in self.__indices_list
    
    def add_index(self, index_name, index_func):
        """Adds the given index value to the dict"""

        if self.compute_index(index_name):
            self.__values[index_name] = index_func() 
        else:
            self.__values[index_name] = "NC"

def process_building(building, building_id,
                     filter_building_id,
                     repair, with_indices,
                     precision, density_2d, density_3d,
                     errors, plot_buildings,
                     vertices, neighbours=[], custom_indices=[]):

    # --- Skip conditions
    # Skip if building ID doesn't match filter when "-f --filter-building-id" is specified
    if filter_building_id and filter_building_id != building_id:
        return building_id, None
    # Skip if type is not Building or Building part
    if building["type"] not in ["Building", "BuildingPart"]:
        return building_id, None
    # Skip if no geometry is found
    if "geometry" not in building or len(building["geometry"]) == 0:
        return building_id, None

    # Geometric variables initialization
    geom = building["geometry"][0]
    boundaries = cityjson.get_surface_boundaries(geom)
    points = cityjson.get_points(geom, vertices)
    
    # --- Mesh creation
    mesh = cityjson.to_polydata(geom, vertices)

    try:
        tri_mesh = cityjson.to_triangulated_polydata(geom, vertices)
    except:
        print(f"{building_id} geometry parsing crashed! Omitting...")
        return building_id, {"type": building["type"]}
    tri_mesh, t = geometry.move_to_origin(tri_mesh)
    fixed = MeshFix(tri_mesh).repair().mesh if repair else tri_mesh
    # Plotting each building if "--plot-buildings" parameter is specified
    if plot_buildings:
        print(f"Plotting {building_id}")
        tri_mesh.plot(show_grid=True)

    # --- Compute surface bearings
    bin_count, bin_edges = get_wall_bearings(mesh, 36)
    xzc, yzc, be = get_roof_bearings(mesh, 36)

    # --- Compute surface properties
    area, point_count, surface_count = geometry.area_by_surface(mesh)
                         
    # --- Compute surface properties
    surface_normals = compute_surface_normal(boundaries, vertices).round(precision)
    surface_azimuths = get_azimuth(surface_normals[:, 0], surface_normals[:, 1]).round(precision)
    surface_inclinations = get_elevation_angle(surface_normals[:, 0], surface_normals[:, 1], surface_normals[:, 2]).round(precision)
    surface_areas = compute_surface_area(mesh).round(precision)
    # Wall properties
    wall_azimuths, _ = list(filter_by_semantic_surface(mesh, surface_azimuths, "WallSurface"))
    wall_inclinations, _ = list(filter_by_semantic_surface(mesh, surface_inclinations, "WallSurface"))
    wall_areas, _ = list(filter_by_semantic_surface(mesh, surface_areas, "WallSurface"))
    # Roof properties
    roof_azimuths, _ = list(filter_by_semantic_surface(mesh, surface_azimuths, "RoofSurface"))
    roof_inclinations, _ = list(filter_by_semantic_surface(mesh, surface_inclinations, "RoofSurface"))
    roof_areas, _ = list(filter_by_semantic_surface(mesh, surface_areas, "RoofSurface"))
    # Roof type properties
    roof_type = np.array(roof_inclinations) > 3 # If the roof inclination is superior to 3 degrees, it is considered 'sloped'
    surface_count["RoofSurfaceSloped"] = sum(roof_type)
    surface_count["RoofSurfaceFlat"] = len(roof_type) - surface_count["RoofSurfaceSloped"] 
    roof_type = np.where(roof_type, "sloped", "flat")

    # --- Height statistics
    roof_points = (geometry.get_points_of_type(mesh, "RoofSurface") if "semantics" in geom else [])
    ground_points = (geometry.get_points_of_type(mesh, "GroundSurface") if "semantics" in geom else [])
    height_stats = (compute_stats([v[2] for v in roof_points]) if roof_points.size > 0 else compute_stats([0]))
    ground_z = (min([v[2] for v in ground_points]) if ground_points.size > 0 else mesh.bounds[4])

    # --- Shape calculations
    shape_2d, shape_3d = cityjson.to_shapely(geom, vertices, ground_only=bool(ground_points.size > 0))
    shape_2d_area = shape_2d.area
                         
    # --- Volume calculations
    aabb_volume = boundingbox_volume(points)
    ch_volume = convexhull_volume(points)

    # --- Bounding box calculations
    obb_2d, _ = cityjson.to_shapely(geom, vertices, ground_only=False)
    obb_2d = obb_2d.minimum_rotated_rectangle
    min_z, max_z = np.min(mesh.clean().points[:, 2]), np.max(mesh.clean().points[:, 2])
    obb = geometry.extrude(obb_2d, min_z, max_z)
    S, L = si.get_box_dimensions(obb_2d)

    # --- 3DBM attribute list
    values = {
        # --- Identifiers
        "building_ID": -1,
        "type": building["type"],
        "lod": geom["lod"],
        # --- Point count
        "point_count": len(points),
        "unique_point_count": fixed.n_points,
        "ground_point_count": point_count["GroundSurface"],
        "wall_point_count": point_count["WallSurface"],
        "roof_point_count": point_count["RoofSurface"],
        # --- Surface count
        "surface_count": len(cityjson.get_surface_boundaries(geom)),
        "ground_surface-count": surface_count["GroundSurface"],
        "wall_surface_count": surface_count["WallSurface"],
        "roof_surface_count": surface_count["RoofSurface"],
        "roof_surface_sloped_count": surface_count["RoofSurfaceSloped"],
        "roof_surface_flat_count": surface_count["RoofSurfaceFlat"],
        # --- Height
        "ground_Z": ground_z,
        "min_Z": height_stats["Min"],
        "max_Z": height_stats["Max"],
        "height_range": height_stats["Range"],
        "mean_Z": height_stats["Mean"],
        "median_Z": height_stats["Median"],
        "mode_Z": height_stats["Mode"] if height_stats["ModeStatus"] == "Y" else "NA",
        "std_Z": height_stats["Std"],
        # --- Area
        "total_surface_area": mesh.area,
        "total_ground_area": area["GroundSurface"],
        "total_wall_area": area["WallSurface"],
        "total_roof_area": area["RoofSurface"],
        "roof_sloped_area": area["RoofSurfaceSloped"],
        "roof_flat_area": area["RoofSurfaceFlat"],
        # --- Volume
        "actual_volume": fixed.volume,
        "convex_hull_volume": ch_volume,
        "obb_volume": obb.volume,
        "aabb_volume": aabb_volume,
        # --- Dimensions
        "footprint_perimeter": shape_2d.length,
        "obb_width": S,
        "obb_length": L,
        # --- Surface lists - area, orientation, inclination
        "surface_areas": str(surface_areas.tolist()),
        "surface_azimuths": str(surface_azimuths.tolist()),
        "surface_inclinations": str(surface_inclinations.tolist()),
        "wall_surface_areas": str(wall_areas.tolist()),
        "wall_surface_azimuths": str(wall_azimuths.tolist()),
        "wall_surface_inclinations": str(wall_inclinations.tolist()),
        "roof_surface_areas": str(roof_areas.tolist()),
        "roof_surface_azimuths": str(roof_azimuths.tolist()),
        "roof_surface_inclinations": str(roof_inclinations.tolist()),
        "roof_surface_types": str(roof_type.tolist()),
        # --- Plot     
        "orientation_values": str(bin_count),
        "orientation_edges": str(bin_edges),
        # --- Errors
        "errors": str(errors),
        "valid": len(errors) == 0,
        "hole_count": tri_mesh.n_open_edges,
        # --- Geometry
        "geometry_2d": shape_2d,
        "geometry_3d": shape_3d
    }

    # Generating 3DBM indices if "-i --with-indices" parameter is specified
    if with_indices:
        voxel = pv.voxelize(tri_mesh, density=density_3d, check_surface=False)
        grid = voxel.cell_centers().points
        grid_point_count = len(grid)
        valid_grid = grid_point_count > 2
        
        shared_area = 0
        closest_distance = float('inf')

        if neighbours:
            # Get neighbour meshes
            for geom in neighbours:
                neighbour_mesh = cityjson.to_triangulated_polydata(geom, vertices).clean()
                neighbour_mesh.points -= t
                n_meshes.append(neighbour_mesh)
            
            # Compute shared walls
            walls = np.hstack([geometry.intersect_surfaces([fixed, neighbour]) for neighbour in n_meshes])
            shared_area = sum([wall["area"][0] for wall in walls])

            # Find the closest distance
            for mesh in n_meshes:
                mesh.compute_implicit_distance(fixed, inplace=True)
                min_distance = np.min(mesh["implicit_distance"])
                closest_distance = min(closest_distance, min_distance)
            
            closest_distance = max(closest_distance, 0)
        else:
            closest_distance = "NA"

        # --- 3DBM indices list
        builder = StatValuesBuilder(values, custom_indices)
        builder.add_index("2d_grid_point_count", lambda: len(si.create_grid_2d(shape_2d, density=density_2d)))
        builder.add_index("3d_grid_point_count", lambda: grid_point_count)
        builder.add_index("circularity_2d", lambda: si.circularity(shape_2d))
        builder.add_index("hemisphericality_3d", lambda: si.hemisphericality(fixed))
        builder.add_index("convexity_2d", lambda: shape_2d_area / shape_2d.convex_hull.area)
        builder.add_index("convexity_3d", lambda: fixed.volume / ch_volume if ch_volume != 0 else 0)
        builder.add_index("fractality_2d", lambda: si.fractality_2d(shape_2d))
        builder.add_index("fractality_3d", lambda: si.fractality_3d(fixed))
        builder.add_index("rectangularity_2d", lambda: shape_2d_area / obb_2d.area)
        builder.add_index("rectangularity_3d", lambda: fixed.volume / obb.volume)
        builder.add_index("squareness_2d", lambda: si.squareness(shape_2d))
        builder.add_index("cubeness_3d", lambda: si.cubeness(fixed))
        builder.add_index("horizontal_elongation", lambda: si.elongation(S, L))
        builder.add_index("min_vertical_elongation", lambda: si.elongation(L, height_stats["Max"]))
        builder.add_index("max_vertical_elongation", lambda: si.elongation(S, height_stats["Max"]))
        builder.add_index("form_factor_3D", lambda: shape_2d_area / math.pow(fixed.volume, 2/3) if fixed.volume != 0 else 0)
        builder.add_index("equivalent_rectangularity_index_2d", lambda: si.equivalent_rectangular_index(shape_2d))
        builder.add_index("equivalent_prism_index_3d", lambda: si.equivalent_prism_index(fixed, obb))
        builder.add_index("proximity_index_2d_", lambda: si.proximity_2d(shape_2d, density=density_2d))
        builder.add_index("proximity_index_3d", lambda: si.proximity_3d(tri_mesh, grid, density=density_3d) if valid_grid else "NA")
        builder.add_index("exchange_index_2d", lambda: si.exchange_2d(shape_2d))
        builder.add_index("exchange_index_3d", lambda: si.exchange_3d(tri_mesh, density=density_3d))
        builder.add_index("spin_index_2d", lambda: si.spin_2d(shape_2d, density=density_2d))
        builder.add_index("spin_index_3d", lambda: si.spin_3d(tri_mesh, grid, density=density_3d) if valid_grid else "NA")
        builder.add_index("perimeter_index_2d", lambda: si.perimeter_index(shape_2d))
        builder.add_index("circumference_index_3d", lambda: si.circumference_index_3d(tri_mesh))
        builder.add_index("depth_index_2d", lambda: si.depth_2d(shape_2d, density=density_2d))
        builder.add_index("depth_index_3d", lambda: si.depth_3d(tri_mesh, density=density_3d) if valid_grid else "NA")
        builder.add_index("girth_index_2d", lambda: si.girth_2d(shape_2d))
        builder.add_index("girth_index_3d", lambda: si.girth_3d(tri_mesh, grid, density=density_3d) if valid_grid else "NA")
        builder.add_index("dispersion_index_2d", lambda: si.dispersion_2d(shape_2d, density=density_2d))
        builder.add_index("dispersion_index_3d", lambda: si.dispersion_3d(tri_mesh, grid, density=density_3d) if valid_grid else "NA")
        builder.add_index("range_index_2d", lambda: si.range_2d(shape_2d))
        builder.add_index("range_index_3d", lambda: si.range_3d(tri_mesh))
        builder.add_index("roughness_index_2d", lambda: si.roughness_index_2d(shape_2d, density=density_2d))
        builder.add_index("roughness_index_3d", lambda: si.roughness_index_3d(tri_mesh, grid, density_2d) if valid_grid else "NA")
        builder.add_index("shared_walls_area", lambda: shared_area)
        builder.add_index("closest_distance", lambda: closest_distance)

    return building_id, values

def city_stats(input,
               filter_lod, filter_building_id,
               repair, with_indices,
               precision, density_2d, density_3d,
               val3dity_report, break_on_error, plot_buildings,
               single_threaded, jobs):
    
    cm = json.load(input) # Load input CityJSON
    original_cm = deepcopy(cm) # Create a deep copy of the input cityjson for later
    filter_level_of_detail(cm, filter_lod) # Filtering out geometries that match the specified LoD (default: 2.2)

    # Transform vertex coordinates back to their original georeferenced values
    if "transform" in cm:
        s = cm["transform"]["scale"]
        t = cm["transform"]["translate"]
        verts = [[v[0] * s[0] + t[0], v[1] * s[1] + t[1], v[2] * s[2] + t[2]]
                for v in cm["vertices"]]
    else:
        verts = cm["vertices"]
    vertices = np.array(verts)

    # Val3dity report
    report = json.load(val3dity_report) if val3dity_report else {}
    if report and not validate_report(report, cm):
        print("This doesn't seem like the right report for this file.")
        return

    # Create R-tree index for the city model
    p = rtree.index.Property()
    p.dimension = 3
    r = rtree.index.Index(tree_generator_function(cm, vertices), properties=p)

    parent_attributes = get_parent_attributes(cm) # Retrieving building attributes

    filter_building_type(cm) # Filtering out objects that match the specified building type (default: BuildingPart)
                   
    stats = {}
    # Single-threaded processing
    if single_threaded or jobs == 1:
        for cityobject_id in tqdm(cm["CityObjects"]):
            cityobject = cm["CityObjects"][cityobject_id]
            # Finding neighbouring buildings
            neighbours = get_neighbours(cm, cityobject_id, r, verts)
            # Retrieving errors for the current building from the validity report
            errors = get_errors_from_report(report, cityobject_id, cm)
            # Generating 3DBM metrics for the current building
            try:
                building_id, vals = process_building(building=cityobject, building_id=cityobject_id,
                                        filter_building_id=filter_building_id,
                                        repair=repair, with_indices=with_indices,
                                        precision=precision, density_2d=density_2d, density_3d=density_3d,
                                        vertices=vertices, neighbours=neighbours,
                                        plot_buildings=plot_buildings, errors=errors)
                # Add autobuilding ID for later joining with original attributes
                if vals is not None:
                    parent_id = building_id.split('-')[0] if '-' in building_id else building_id
                    vals["building_ID"] = parent_id
                    stats[building_id] = vals

            except Exception as e:
                print(f"Problem with {building_id}")
                if break_on_error:
                    raise e

    else:
        from concurrent.futures import ProcessPoolExecutor

        num_objs = len(cm["CityObjects"])
        num_cores = jobs

        with ProcessPoolExecutor(max_workers=num_cores) as pool:
            with tqdm(total=num_objs) as progress:
                futures = []

                for cityobject_id in cm["CityObjects"]:
                    cityobject = cm["CityObjects"][cityobject_id]
                    # Finding neighbouring buildings
                    neighbours = get_neighbours(cm, cityobject_id, r, verts)
                    # Retrieving errors for the current building from the validity report
                    errors = get_errors_from_report(report, cityobject_id, cm)
                    # Generating 3DBM metrics for the current building
                    future = pool.submit(process_building,
                                         building=cityobject, building_id=cityobject_id,
                                         filter_building_id=filter_building_id,
                                         repair=repair, with_indices=with_indices,
                                         precision=precision, density_2d=density_2d, density_3d=density_3d,
                                         vertices=vertices, neighbours=neighbours,
                                         plot_buildings=plot_buildings, errors=errors)
                    
                    future.add_done_callback(lambda p: progress.update())
                    futures.append(future)
                
                for future in futures:
                    try:
                        building_id, vals = future.result()
                        # Add autobuilding ID for later joining with original attributes
                        if vals is not None:
                            parent_id = building_id.split('-')[0] if '-' in building_id else building_id
                            vals["building_ID"] = parent_id
                            stats[building_id] = vals
    
                    except Exception as e:
                        print(f"Problem with {cityobject_id}")
                        if break_on_error:
                            raise e

    # DataFrame creation
    click.echo("Building data frame...")

    df_original_attributes = pd.DataFrame.from_dict(parent_attributes, orient="index").round(precision)
    df_3dbm_attributes = pd.DataFrame.from_dict(stats, orient="index").round(precision)
    # Combine original attributes and 3DBM attributes together
    merged_df = pd.merge(df_original_attributes.reset_index(), df_3dbm_attributes.reset_index(), left_on='index', right_on='building_ID', how='inner')
    merged_df = merged_df.drop(columns=['index_x', 'index_y'])

    return merged_df, original_cm

def export_to_gpkg(gdf, output_gpkg, geometry_col, crs):
    gdf = gpd.GeoDataFrame(gdf, geometry=geometry_col)

    other_geometry_col = 'geometry_3d' if geometry_col == 'geometry_2d' else 'geometry_2d'
    gdf = gdf.drop(columns=[other_geometry_col], errors='ignore')
    
    columns_to_drop = ['surface_areas', 'surface_azimuths', 'surface_inclinations', 
                       'wall_surface_areas', 'wall_surface_azimuths', 'wall_surface_inclinations',
                       'roof_surface_areas', 'roof_surface_azimuths', 'roof_surface_inclinations', 'roof_surface_types']
    
    gdf = gdf.drop(columns=[col for col in columns_to_drop if col in gdf.columns])
    output_file = output_gpkg.split(".")[0] + f"_{geometry_col.upper()}.gpkg"
    
    gdf.to_file(output_file, crs=crs, driver="GPKG", engine="fiona")

def process_files(input, output_cityjson, output_csv, output_gpkg,
                  filter_lod, filter_building_id,
                  repair, with_indices,
                  precision, density_2d, density_3d,
                  val3dity_report, break_on_error, plot_buildings,
                  single_threaded, jobs):

    df, cm = city_stats(input=input,
                        filter_lod=filter_lod, filter_building_id=filter_building_id,
                        repair=repair, with_indices=with_indices,
                        precision=precision, density_2d=density_2d, density_3d=density_3d,
                        val3dity_report=val3dity_report, break_on_error=break_on_error, plot_buildings=plot_buildings,
                        single_threaded=single_threaded, jobs=jobs)

    # Retrieving the CRS from the input CityJSON
    crs = cm["metadata"]["referenceSystem"].split('/')[-1]
    if "+" in crs:
        crs = "EPSG:" + "+EPSG:".join(crs.split('+'))

    # Export into GPKG if "-g --output-gpkg" is specified
    if output_gpkg:
        # Export 2D GPKG
        export_to_gpkg(df, output_gpkg, 'geometry_2d', crs)
        # Export 3D GPKG
        export_to_gpkg(df, output_gpkg, 'geometry_3d', crs)

    drop_columns = ['geometry_2d','geometry_3d',
                    'wall_surface_areas', 'wall_surface_azimuths', 'wall_surface_inclinations',
                    'roof_surface_areas', 'roof_surface_azimuths', 'roof_surface_inclinations', 'roof_surface_types']
    df = df.drop(columns=drop_columns)

    # Export into CSV if "-c --output-csv" is specified
    if output_csv:
        click.echo("Writing output...")
        df.to_csv(output_csv, index=False)

    df[['surface_areas', 'surface_azimuths', 'surface_inclinations']] = df[['surface_areas', 'surface_azimuths', 'surface_inclinations']].applymap(ast.literal_eval)
                      
    # Export into CityJSON if "-o --output-cityjson" is specified
    if output_cityjson:
        city_objects = cm["CityObjects"]
        for index, row in df.iterrows():
            building_part_id = str(row["building_ID"]) + "-0"
            building_id = row["building_ID"]
            lod = row["lod"]
            
            cityobject = city_objects[building_id]
            cityobject_part = city_objects[building_part_id]
            
            geometry_part = next(geom for geom in cityobject_part["geometry"] if str(geom["lod"]) == lod)
            
            geometry_part["semantics"]["+areas"] = [row["surface_areas"]]
            geometry_part["semantics"]["+azimuths"] = [row["surface_azimuths"]]
            geometry_part["semantics"]["+inclinations"] = [row["surface_inclinations"]]

            row = row.drop(['surface_areas', 'surface_azimuths', 'surface_inclinations'])
            
            cityobject["attributes"] = row.to_dict()

        with open(output_cityjson, 'w') as out_file:
            json.dump(cm, out_file)

# Parameters
@click.command()
@click.argument("input", type=click.File("rb"))
@click.option('-c', '--output-csv', type=click.File("wb"))
@click.option('-o', '--output-cityjson')
@click.option('-g', '--output-gpkg')
@click.option('-l', '--filter-lod', default = '2.2')
@click.option('-f', '--filter-building-id', default=None)
@click.option('-r', '--repair', flag_value=True)
@click.option('-i', '--with-indices', flag_value=True)
@click.option('--precision', default = 2)
@click.option('--density-2d', default=1.0)
@click.option('--density-3d', default=1.0)
@click.option('-v', '--val3dity-report', type=click.File("rb"))
@click.option('-b', '--break-on-error', flag_value=True)
@click.option('-p', '--plot-buildings', flag_value=True)
@click.option('-s', '--single-threaded', flag_value=True)
@click.option('-j', '--jobs', default=1)
def main(input, output_cityjson, output_csv, output_gpkg,
         filter_lod, filter_building_id,
         repair, with_indices,
         precision, density_2d, density_3d,
         val3dity_report, break_on_error, plot_buildings,
         single_threaded, jobs):

    process_files(input=input, output_cityjson=output_cityjson, output_csv=output_csv, output_gpkg=output_gpkg,
                  filter_lod=filter_lod, filter_building_id=filter_building_id,
                  repair=repair, with_indices=with_indices,
                  precision=precision, density_2d=density_2d, density_3d=density_3d,
                  val3dity_report=val3dity_report, break_on_error=break_on_error, plot_buildings=plot_buildings,
                  single_threaded=single_threaded, jobs=jobs)

if __name__ == "__main__":
    main()
