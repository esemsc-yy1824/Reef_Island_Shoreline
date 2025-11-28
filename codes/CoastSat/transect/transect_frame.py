import geopandas as gpd
from shapely.geometry import LineString, Point
import numpy as np
import os

def generate_radial_transects(output, filepath, sitename, settings, year=2016, length=300):
    """
    Generate radial transects from the shoreline centroid.

    Args:
        output (dict): CoastSat shoreline extraction results.
        filepath (str): Output directory path.
        sitename (str): Output file name prefix.
        settings (dict): Configuration with projection info (e.g., EPSG).
        year (int, optional): Target year for shoreline selection. Defaults to 2016.
        length (float, optional): Length of each transect. Defaults to 300.

    Saves:
        GeoJSON file containing radial transects.
    """
    shorelines = [sl for sl, dt in zip(output['shorelines'], output['dates']) if dt.year == year]
    shoreline_coords = shorelines[0]
    shoreline_line = LineString(shoreline_coords)
    centroid = shoreline_line.centroid

    dists = [Point(x, y).distance(centroid) for x, y in shoreline_coords]
    R_mean = np.mean(dists)

    s = 50
    delta_theta = s / R_mean
    n_transects = int(np.ceil(2 * np.pi / delta_theta))

    transects = []
    angles_deg = np.linspace(0, 360, n_transects, endpoint=False)

    for angle in angles_deg:
        theta_rad = np.deg2rad(angle)
        end_x = centroid.x + np.cos(theta_rad) * length
        end_y = centroid.y + np.sin(theta_rad) * length
        transect = LineString([(centroid.x, centroid.y), (end_x, end_y)])
        transects.append(transect)

    transect_gdf = gpd.GeoDataFrame(
        {'name': [f'T{i+1}' for i in range(len(transects))]},
        geometry=transects,
        crs=f"EPSG:{settings['output_epsg']}"
    )

    outpath = os.path.join(filepath, sitename + '_radial_transects.geojson')
    transect_gdf.to_file(outpath, driver='GeoJSON', encoding='utf-8')

    print(f"Radial transects saved to: {outpath}")


def generate_hybrid_transects(output, filepath, sitename, settings, window_size, year, length):
    """
    Generate transects perpendicular to the shoreline while ignoring small surrounding islands.

    Args:
        output (dict): CoastSat shoreline extraction results.
        filepath (str): Output directory path.
        sitename (str): Output file name prefix.
        settings (dict): Configuration with projection info (e.g., EPSG).
        window_size (int): Number of points for tangent estimation (larger = smoother).
        year (int): Target year for shoreline selection.
        length (float): Length of each transect.

    Saves:
        GeoJSON file containing transects.
    """
    shorelines_2016 = [sl for sl, dt in zip(output['shorelines'], output['dates']) if dt.year == year]
    shoreline_coords = shorelines_2016[0]
    shoreline_line = LineString(shoreline_coords)
    centroid = shoreline_line.centroid

    dists = [Point(x, y).distance(centroid) for x, y in shoreline_coords]
    R_mean = np.mean(dists)

    s = 50
    delta_theta = s / R_mean
    n_transects = int(np.ceil(2 * np.pi / delta_theta))
    angles_deg = np.linspace(0, 360, n_transects, endpoint=False)

    transects = []

    coords_array = np.array(shoreline_coords)
    n_coords = len(coords_array)

    for angle in angles_deg:
        theta_rad = np.deg2rad(angle)

        end_x = centroid.x + np.cos(theta_rad) * 10000
        end_y = centroid.y + np.sin(theta_rad) * 10000
        radial_line = LineString([(centroid.x, centroid.y), (end_x, end_y)])

        intersection = radial_line.intersection(shoreline_line)
        if intersection.is_empty:
            continue

        if intersection.geom_type == 'MultiPoint':
            intersection_pt = min(intersection.geoms, key=lambda p: p.distance(centroid))
        elif intersection.geom_type == 'LineString':
            intersection_pt = intersection.interpolate(0.5, normalized=True)
        elif intersection.geom_type == 'Point':
            intersection_pt = intersection
        else:
            continue

        closest_idx = np.argmin([Point(x, y).distance(intersection_pt) for x, y in shoreline_coords])

        half_window = window_size // 2
        idx_range = [(closest_idx + i) % n_coords for i in range(-half_window, half_window + 1)]
        pts_window = coords_array[idx_range]

        x = pts_window[:, 0]
        y = pts_window[:, 1]
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]

        dx = 1
        dy = m
        norm = np.hypot(dx, dy)
        if norm == 0:
            continue
        dx /= norm
        dy /= norm

        nx, ny = dy, -dx

        tx1 = intersection_pt.x + nx * length / 2
        ty1 = intersection_pt.y + ny * length / 2
        tx2 = intersection_pt.x - nx * length / 2
        ty2 = intersection_pt.y - ny * length / 2

        p1 = Point(tx1, ty1)
        p2 = Point(tx2, ty2)
        if p1.distance(centroid) < p2.distance(centroid):
            start, end = (tx1, ty1), (tx2, ty2)
        else:
            start, end = (tx2, ty2), (tx1, ty1)

        transect = LineString([start, end])
        transects.append(transect)

    transect_gdf = gpd.GeoDataFrame(
        {'name': [f'T{i + 1}' for i in range(len(transects))]},
        geometry=transects,
        crs=f"EPSG:{settings['output_epsg']}"
    )

    outpath = os.path.join(filepath, sitename + '_hybrid_transects.geojson')
    transect_gdf.to_file(outpath, driver='GeoJSON', encoding='utf-8')

    print(f"Hybrid transects saved to: {outpath}")