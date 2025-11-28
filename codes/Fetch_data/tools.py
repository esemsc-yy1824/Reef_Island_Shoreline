import os
import glob
import warnings
from math import atan2, degrees
from datetime import datetime

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.transform import Affine
from shapely.geometry import box, Polygon
from pyproj import Transformer


warnings.filterwarnings("ignore", category=RuntimeWarning)


def export_shoreline_coords(sitename, transect_types='hybrid'):
    """
    Export shoreline intersection coordinates based on transect type and time series data.

    Saves:
        CSV file containing shoreline intersection coordinates (UTM X/Y).
    """
    if transect_types == 'radial':
        transects_gdf = gpd.read_file(f"../CoastSat/data/{sitename}/{sitename}_radial_transects.geojson")
    elif transect_types == 'hybrid':
        transects_gdf = gpd.read_file(f"../CoastSat/data/{sitename}/{sitename}_hybrid_transects.geojson")
    else:
        raise ValueError("transect_types must be 'radial' or 'hybrid'")

    transect_dict = {}
    for idx, row in transects_gdf.iterrows():
        transect_id = f"T{idx+1}"
        line = row.geometry
        x0, y0 = line.coords[0]
        x1, y1 = line.coords[-1]
        transect_dict[transect_id] = {'start': (x0, y0), 'end': (x1, y1)}

    df_time_series = pd.read_csv(f"../CoastSat/data/{sitename}/transect_time_series_tidally_corrected.csv")

    results = []
    for _, row in df_time_series.iterrows():
        date = row['dates']
        for transect_id, coords in transect_dict.items():
            dist = row[transect_id]
            start_x, start_y = coords['start']
            end_x, end_y = coords['end']

            dx = end_x - start_x
            dy = end_y - start_y
            length = np.hypot(dx, dy)
            if length == 0:
                intersection_x = np.nan
                intersection_y = np.nan
            else:
                ux = dx / length
                uy = dy / length
                intersection_x = start_x + dist * ux
                intersection_y = start_y + dist * uy

            results.append({
                'dates': date,
                'transect_id': transect_id,
                'longitude': intersection_x,
                'latitude': intersection_y
            })

    df_result = pd.DataFrame(results)

    out_dir = f"./Model_Data_{sitename}/"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print(f"Created folder: Model_Data_{sitename}")

    df_result['longitude'] = pd.to_numeric(df_result['longitude'], errors='coerce')
    df_result['latitude'] = pd.to_numeric(df_result['latitude'], errors='coerce')
    df_result_clean = df_result.dropna(subset=['longitude', 'latitude'])
    df_result_clean.to_csv(os.path.join(out_dir, "shoreline_coords.csv"), index=False)
    print(f"Shoreline coordinates saved to {out_dir}shoreline_coords.csv")


# def compute_shoreline_slope(sitename, transects_gdf,
#                             dem_path="nasadem.tif",
#                             data_root="../CoastSat/data",
#                             output_root="./Model_Data",
#                             window=40, step=5):
#     """
#     Compute shoreline slope time series from transects and DEM.

#     Returns:
#     df_slope : DataFrame
#         Long-format DataFrame with columns [date, transect_id, slope_m_per_m].
#     """

#     # transformer for projection
#     transformer = Transformer.from_crs("EPSG:32643", "EPSG:4326", always_xy=True)

#     # DEM info
#     with rasterio.open(dem_path) as src:
#         dem_bounds = src.bounds

#     # shoreline time series
#     # csv_path = f"{data_root}/{sitename}/transect_time_series_tidally_corrected.csv"
#     csv_path = f"{data_root}/{sitename}/{sitename}_Delta_d_p.csv"

#     df_cross = pd.read_csv(csv_path, index_col=0)

#     dates = df_cross["dates"]
#     transect_names = [col for col in df_cross.columns if col != "dates"]

#     records = []

#     for transect_id in transect_names:
#         match = transects_gdf.loc[transects_gdf["name"] == transect_id, "geometry"]

#         if match.empty:
#             print(f"Transect {transect_id} not found in geojson.")
#             continue

#         line = match.iloc[0]

#         for i, date in enumerate(dates):
#             shoreline_pos = df_cross.loc[i, transect_id]

#             if np.isnan(shoreline_pos):
#                 continue

#             if shoreline_pos <= window:
#                 print(f"Shoreline {shoreline_pos:.2f} too close for {transect_id} on {date}.")
#                 continue

#             start_dist = shoreline_pos - window
#             end_dist   = shoreline_pos
#             distances = np.arange(start_dist, end_dist + step, step)

#             # sample points
#             points_along = [line.interpolate(d) for d in distances]

#             coords_kept, distances_kept = [], []
#             for d, p in zip(distances, points_along):
#                 lon, lat = transformer.transform(p.x, p.y)
#                 if (dem_bounds.left <= lon <= dem_bounds.right) and \
#                    (dem_bounds.bottom <= lat <= dem_bounds.top):
#                     coords_kept.append((lon, lat))
#                     distances_kept.append(d)

#             if len(coords_kept) < 2:
#                 print(f" Not enough valid points for {transect_id} on {date}.")
#                 continue

#             coords_along = coords_kept
#             distances_along = np.array(distances_kept)

#             with rasterio.open(dem_path) as src:
#                 elevs = [list(src.sample([pt]))[0][0] for pt in coords_along]

#             elevs = np.array(elevs)
#             valid_idx = elevs != -32768

#             if valid_idx.sum() < 2:
#                 continue

#             if np.all(elevs[valid_idx] == elevs[valid_idx][0]):
#                 slope_m_per_m = 0.0
#             else:
#                 dz = elevs[valid_idx].max() - elevs[valid_idx].min()
#                 dx = (len(distances_along[valid_idx]) - 1) * step
#                 slope_m_per_m = dz / dx if dx > 0 else 0.0

#             records.append({
#                 "date": date,
#                 "transect_id": transect_id,
#                 "slope_m_per_m": slope_m_per_m
#             })

#     df_slope = pd.DataFrame(records)

#     # save
#     outdir = f"{output_root}_{sitename}"
#     os.makedirs(outdir, exist_ok=True)
#     outpath = f"{outdir}/shoreline_slope.csv"
#     df_slope["date"] = pd.to_datetime(df_slope["date"], utc=True)
#     df_slope.to_csv(outpath, index=False)

#     print(f"Slope timeseries CSV saved to {outpath}")
#     return df_slope

def compute_shoreline_slope(sitename, transects_gdf,
                            dem_path="nasadem.tif",
                            data_root="../CoastSat/data",
                            output_root="./Model_Data",
                            window=40, step=5,
                            use_normalized=False,  # Set True to support Δd′, see comments below
                            baseline_year=2016):
    """
    Compute the terrain slope (m/m) near the shoreline within a given window 
    for each transect and each date.

    - The line.interpolate() sampling must use "meters" as the distance unit.  
    - By default, it reads the tide-corrected absolute distance CSV: transect_time_series_tidally_corrected.csv.  
    - If use_normalized=True, it reads {sitename}_Delta_d_p.csv for Δd′ and converts it back 
      to meters internally using baseline + R_mean.
    """

    # Open DEM once
    with rasterio.open(dem_path) as src:
        dem_bounds = src.bounds
        dem_crs = src.crs

        # Dynamically create a CRS transformer from transects CRS to DEM CRS (do not hard-code EPSG)
        if transects_gdf.crs is None:
            raise ValueError("transects_gdf.crs is empty. Please make sure the geojson has CRS defined.")
        transformer = Transformer.from_crs(transects_gdf.crs, dem_crs, always_xy=True)

        # Read timeseries (prefer tide-corrected absolute distances)
        if not use_normalized:
            csv_path = f"{data_root}/{sitename}/transect_time_series_tidally_corrected.csv"
            df_cross = pd.read_csv(csv_path)
            # Remove unnamed columns
            df_cross = df_cross.loc[:, ~df_cross.columns.str.contains(r'^Unnamed', case=False)]
        else:
            # If only Δd′ file is kept, restore it back to meters (optional)
            ddp_path = f"{data_root}/{sitename}/{sitename}_Delta_d_p.csv"
            df_ddp = pd.read_csv(ddp_path)
            df_ddp = df_ddp.loc[:, ~df_ddp.columns.str.contains(r'^Unnamed', case=False)]
            if 'dates' not in df_ddp.columns:
                raise ValueError("'dates' column is missing in Delta_d_p CSV.")

            # Read tide-corrected file to get baseline (d_base) and R_mean
            corr_path = f"{data_root}/{sitename}/transect_time_series_tidally_corrected.csv"
            df_corr = pd.read_csv(corr_path)
            df_corr = df_corr.loc[:, ~df_corr.columns.str.contains(r'^Unnamed', case=False)]

            # Align T columns and sort
            t_cols = [c for c in df_ddp.columns if c.upper().startswith('T')]
            t_cols = sorted(t_cols, key=lambda c: int(''.join(filter(str.isdigit, c)) or 0))
            t_cols = [c for c in t_cols if c in df_corr.columns]

            # Compute baseline index and R_mean using tide-corrected matrix
            dates_corr = pd.to_datetime(df_corr['dates'], errors='coerce')
            M_corr = df_corr[t_cols].to_numpy(float).T  # (n_transects, n_times)

            years = dates_corr.dt.year.to_numpy()
            baseline_idx = np.full(M_corr.shape[0], -1, dtype=int)
            for ti in range(M_corr.shape[0]):
                idxs = np.where((years == baseline_year) & ~np.isnan(M_corr[ti]))[0]
                if idxs.size == 0:
                    idxs = np.where(~np.isnan(M_corr[ti]))[0]
                if idxs.size > 0:
                    baseline_idx[ti] = idxs.min()

            d_base = np.array([M_corr[i, bi] if bi >= 0 else np.nan
                               for i, bi in enumerate(baseline_idx)], dtype=float)
            R_mean = float(np.nanmedian(d_base))

            # Convert Δd′ back to meters: d = d_base + Δd′ * R_mean
            M_ddp = df_ddp[t_cols].to_numpy(float)          # (n_times, n_transects)
            meters = M_ddp * R_mean + d_base[np.newaxis, :] # (n_times, n_transects)

            df_cross = pd.DataFrame(meters, columns=t_cols)
            df_cross.insert(0, 'dates', df_ddp['dates'])

        # Dates & transects
        if 'dates' not in df_cross.columns:
            raise ValueError("CSV is missing 'dates' column.")
        dates = df_cross["dates"]
        transect_names = [col for col in df_cross.columns if col != "dates"]

        records = []

        # Main loop: compute slope for each transect & date
        for transect_id in transect_names:
            match = transects_gdf.loc[transects_gdf["name"] == transect_id, "geometry"]
            if match.empty:
                print(f"Transect {transect_id} not found in geojson.")
                continue

            line = match.iloc[0]  # Geometry CRS = transects_gdf.crs (projected coordinates, meters)

            for i, date in enumerate(dates):
                shoreline_pos = df_cross.loc[i, transect_id]  # Units: meters (already ensured)

                if np.isnan(shoreline_pos):
                    continue
                if shoreline_pos <= window:
                    # Shoreline is too close to the start point, cannot extract a full window landward
                    # Optionally, you could extend seaward to keep total length
                    print(f"Shoreline {shoreline_pos:.2f} too close for {transect_id} on {date}.")
                    continue

                start_dist = shoreline_pos - window
                end_dist   = shoreline_pos
                distances = np.arange(start_dist, end_dist + step, step)

                # Interpolate points along the line in meters, then transform to DEM CRS
                points_along = [line.interpolate(d) for d in distances]

                coords_kept, distances_kept = [], []
                for d, p in zip(distances, points_along):
                    # Transform to DEM CRS (NASADEM is usually EPSG:4326, lat/lon)
                    x_dem, y_dem = transformer.transform(p.x, p.y)
                    if (dem_bounds.left <= x_dem <= dem_bounds.right) and \
                       (dem_bounds.bottom <= y_dem <= dem_bounds.top):
                        coords_kept.append((x_dem, y_dem))
                        distances_kept.append(d)

                if len(coords_kept) < 2:
                    print(f"Not enough valid points for {transect_id} on {date}.")
                    continue

                distances_along = np.array(distances_kept)

                # Sample elevation values (coords already in DEM CRS)
                elevs = [list(src.sample([pt]))[0][0] for pt in coords_kept]
                elevs = np.array(elevs)

                # Filter invalid DEM values (NASADEM nodata usually -32768)
                nodata = src.nodata
                if nodata is None:
                    nodata = -32768
                valid_idx = elevs != nodata
                if valid_idx.sum() < 2:
                    continue

                # Estimate slope: (max elevation - min elevation) / horizontal length
                dz = elevs[valid_idx].max() - elevs[valid_idx].min()
                dx = (len(distances_along[valid_idx]) - 1) * step
                slope_m_per_m = (dz / dx) if dx > 0 else 0.0

                records.append({
                    "date": date,
                    "transect_id": transect_id,
                    "slope_m_per_m": float(slope_m_per_m)
                })

    # Save output
    df_slope = pd.DataFrame(records)
    outdir = f"{output_root}_{sitename}"
    os.makedirs(outdir, exist_ok=True)
    outpath = f"{outdir}/shoreline_slope.csv"
    df_slope["date"] = pd.to_datetime(df_slope["date"], utc=True, errors='coerce')
    df_slope.to_csv(outpath, index=False)

    print(f"Slope timeseries CSV saved to {outpath}")
    return df_slope


def compute_polygon_orientation(coords: np.ndarray) -> float:
    """
    Compute orientation of a polygon using PCA.
    Returns angle in degrees (0–180).
    """
    coords_centered = coords - coords.mean(axis=0)
    cov = np.cov(coords_centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    principal_vector = eigvecs[:, np.argmax(eigvals)]
    angle_rad = np.arctan2(principal_vector[1], principal_vector[0])
    angle_deg = np.degrees(angle_rad)
    if angle_deg < 0:
        angle_deg += 180
    return angle_deg


def process_shoreline(sl, date, shoreline_coords: pd.DataFrame, records: list):
    """
    Process one shoreline polygon for a given date and append results to records.
    Only keep island_orientation for each date and transect_id.
    """
    try:
        valid_poly = True
        try:
            poly = Polygon(sl)
            if not poly.is_valid or poly.area == 0:
                valid_poly = False
        except Exception:
            valid_poly = False

        if valid_poly:
            island_orientation = compute_polygon_orientation(np.array(sl))
        else:
            island_orientation = np.nan

        df_date = shoreline_coords[shoreline_coords['dates'] == date.strftime("%Y-%m-%d")]
        df_date = df_date.sort_values(by='transect_id').reset_index(drop=True)

        for i in range(len(df_date)):
            records.append({
                "date": date.strftime("%Y-%m-%d"),
                "transect_id": f"T{i+1}",
                "island_orientation": island_orientation
            })

    except Exception as e:
        print(f"Error at {date.strftime('%Y-%m-%d')}: {e}")


def compute_polygon_orientation(coords):
    """
    Compute the main orientation of a polygon using PCA.
    Returns an angle in degrees (0–180).
    """
    # Center the coordinates
    coords_centered = coords - coords.mean(axis=0)
    # Covariance matrix
    cov = np.cov(coords_centered.T)
    # Eigen decomposition
    eigvals, eigvecs = np.linalg.eigh(cov)
    # Select the eigenvector with the largest eigenvalue (principal axis)
    principal_vector = eigvecs[:, np.argmax(eigvals)]
    # Compute angle
    angle_rad = np.arctan2(principal_vector[1], principal_vector[0])
    angle_deg = np.degrees(angle_rad)
    # Ensure angle is between 0 and 180
    if angle_deg < 0:
        angle_deg += 180
    return angle_deg

def build_transect_island_orientation(output, sitename, shoreline_coords_path=None, save=True):
    """
    Build a DataFrame containing only:
    - date
    - transect_id
    - island_orientation
    """
    # Load shoreline_coords
    if shoreline_coords_path is None:
        shoreline_coords_path = f'./Model_Data_{sitename}/shoreline_coords.csv'
    shoreline_coords = pd.read_csv(shoreline_coords_path)

    # Standardize the dates column to YYYY-MM-DD strings
    shoreline_coords['dates'] = pd.to_datetime(shoreline_coords['dates'], errors='coerce')
    shoreline_coords['dates'] = shoreline_coords['dates'].dt.strftime('%Y-%m-%d')

    records = []

    # Iterate over shorelines and their dates
    for sl, date in zip(output["shorelines"], output["dates"]):
        try:
            # Check polygon validity
            valid_poly = True
            try:
                poly = Polygon(sl)
                if not poly.is_valid or poly.area == 0:
                    valid_poly = False
            except Exception:
                valid_poly = False

            # Compute island orientation (or NaN if invalid)
            if valid_poly:
                island_orientation = compute_polygon_orientation(np.array(sl))
            else:
                island_orientation = np.nan

            # Get transects for the current date
            date_str = date.strftime("%Y-%m-%d")
            df_date = shoreline_coords[shoreline_coords['dates'] == date_str]
            df_date = df_date.sort_values(by='transect_id').reset_index(drop=True)

            # Create one record per transect_id
            for i in range(len(df_date)):
                records.append({
                    "date": date_str,
                    "transect_id": f"T{i+1}",
                    "island_orientation": island_orientation
                })

        except Exception as e:
            print(f"❌ Error at {date.strftime('%Y-%m-%d')}: {e}")
            continue

    # Build final DataFrame
    df = pd.DataFrame(records)

    if save:
        out_path = f"./Model_Data_{sitename}/transects_island_orientation.csv"
        df.to_csv(out_path, index=False)
        print(f"Saved transects_island_orientation.csv -> {out_path}")

    return df


def compute_ndvi_by_transect(
    sitename,
    lon_min, lat_min, lon_max, lat_max,
    tolerance_days=5,
    verbose=False,
    data_dir=None,
    shoreline_coords_path=None,
    output_path=None,
    save=True,
):
    """
    Build an NDVI-by-transect table by matching each shoreline date with the
    nearest available satellite image (within ±tolerance_days), computing mean
    NDVI over a fixed geographic bounding box, and assigning that NDVI to all
    transects for that date.

    Returns:
    ndvi_df : pandas.DataFrame
        Table with columns ['date', 'transect_id', 'NDVI'].
    summary : dict
        Coverage summary: {'total_dates': int, 'valid_dates': int, 'coverage_pct': float}
    """
    if data_dir is None:
        data_dir = f'../CoastSat/data/{sitename}'
    if shoreline_coords_path is None:
        shoreline_coords_path = f'./Model_Data_{sitename}/shoreline_coords.csv'
    if output_path is None:
        output_path = f'./Model_Data_{sitename}/ndvi.csv'

    # Build bbox geometry in WGS84
    bbox_geom = box(lon_min, lat_min, lon_max, lat_max)
    bbox_gdf = gpd.GeoDataFrame({'geometry': [bbox_geom]}, crs='EPSG:4326')

    shoreline_coords = pd.read_csv(shoreline_coords_path)
    shoreline_coords['dates'] = pd.to_datetime(shoreline_coords['dates'], errors='coerce')
    shoreline_coords['dates'] = shoreline_coords['dates'].dt.strftime('%Y-%m-%d')
    unique_dates = shoreline_coords['dates'].unique()

    image_files = []
    for sensor in ['L8', 'L9', 'S2']:
        image_files += glob.glob(os.path.join(data_dir, sensor, 'ms', '*.tif'))

    print(f"Found {len(image_files)} image files")

    # Pre-parse image dates from filename prefix (first 10 chars)
    image_date_map = {}
    for img_file in image_files:
        basename = os.path.basename(img_file)
        date_str = basename[:10]  # assumes YYYY-MM-DD at the beginning
        try:
            img_date = datetime.strptime(date_str, '%Y-%m-%d')
            image_date_map[img_file] = img_date
        except ValueError:
            continue

    ndvi_date_lookup = {}

    # Match each shoreline date to nearest image
    for date_str in unique_dates:
        target_date = datetime.strptime(date_str, '%Y-%m-%d')
        min_diff = None
        matched_file = None

        for img_file, img_date in image_date_map.items():
            diff_days = abs((img_date - target_date).days)
            if (min_diff is None) or (diff_days < min_diff):
                min_diff = diff_days
                matched_file = img_file

        if matched_file and min_diff <= tolerance_days:
            if verbose:
                print(f"Using image {os.path.basename(matched_file)} for date {date_str} (diff={min_diff} days)")
        else:
            if verbose:
                print(f"⚠️ No image within ±{tolerance_days} days for {date_str}, NDVI=NaN")
            ndvi_date_lookup[date_str] = np.nan
            continue

        # Compute NDVI over the bbox for the matched image
        try:
            with rasterio.open(matched_file) as src:
                img_crs = src.crs
                # Reproject bbox to the image CRS
                bbox_proj = bbox_gdf.to_crs(img_crs)

                # Crop the image to the bbox
                out_image, out_transform = mask(src, bbox_proj.geometry, crop=True)

                # Band selection (zero-based indices) — UNCHANGED
                if ('L8' in matched_file) or ('L9' in matched_file):
                    red = out_image[3].astype('float32')   # band 4 (0-based idx 3)
                    nir = out_image[4].astype('float32')   # band 5 (0-based idx 4)
                elif 'S2' in matched_file:
                    red = out_image[3].astype('float32')   # band 4 (0-based idx 3)
                    nir = out_image[7].astype('float32')   # band 8 (0-based idx 7)
                else:
                    raise ValueError(f"Unknown satellite type for file: {matched_file}")

                # NDVI computation (with small epsilon) — UNCHANGED
                ndvi = (nir - red) / (nir + red + 1e-6)
                ndvi_mean = np.nanmean(ndvi)

                ndvi_date_lookup[date_str] = ndvi_mean

        except Exception as e:
            # Keep silent (as in original) and assign NaN
            ndvi_date_lookup[date_str] = np.nan

    # Build records by inheriting transect_id and date (unchanged idea)
    records = []
    for _, row in shoreline_coords.iterrows():
        date = row['dates']
        transect_id = row['transect_id']
        ndvi_val = ndvi_date_lookup.get(date, np.nan)
        records.append({'date': date, 'transect_id': transect_id, 'NDVI': ndvi_val})

    ndvi_df = pd.DataFrame(records)

    # Save and report coverage
    if save:
        ndvi_df.to_csv(output_path, index=False)
        print(f"✅ Saved NDVI-by-transect file with ±{tolerance_days} days tolerance: {output_path}")

    # Coverage: unique dates only, since NDVI is shared per date
    ndvi_by_date = ndvi_df[['date', 'NDVI']].drop_duplicates()
    total_dates = len(ndvi_by_date)
    valid_dates = ndvi_by_date['NDVI'].notna().sum()
    coverage = valid_dates / total_dates * 100 if total_dates > 0 else 0.0

    print(f"✅ NDVI coverage: {valid_dates}/{total_dates} dates ({coverage:.2f}%) have valid NDVI values.")

    summary = {
        'total_dates': total_dates,
        'valid_dates': valid_dates,
        'coverage_pct': coverage
    }
    return ndvi_df, summary



def add_monsoon_flag(df, start_month=4, end_month=10, col_name='is_northeast_monsoon'):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['month'] = df['date'].dt.month
    df[col_name] = df['month'].apply(lambda m: 1 if (start_month <= m <= end_month) else 0)
    return df

def build_final_dataset(
    sitename: str,
    transect_types: str,
    cols_to_fill=('slope_m_per_m', 'island_orientation', 'NDVI'),
    monsoon_start=4,
    monsoon_end=10,
    output_path=None,
    save=True,
):
    df_ecmwf = pd.read_csv(f"./Model_Data_{sitename}/ecmwf_transects_timeseries_{transect_types}.csv")
    df_reef  = pd.read_csv(f"./Model_Data_{sitename}/Reef_Geomorphometrics_{transect_types}.csv")
    df_slope = pd.read_csv(f"./Model_Data_{sitename}/shoreline_slope.csv")
    df_geom  = pd.read_csv(f"./Model_Data_{sitename}/transects_island_orientation.csv")
    df_ndvi  = pd.read_csv(f"./Model_Data_{sitename}/ndvi.csv")
    # df_target = pd.read_csv(f"../CoastSat/data/{sitename}/transect_time_series_tidally_corrected.csv")
    df_target = pd.read_csv(f"../CoastSat/data/{sitename}/{sitename}_Delta_d_p.csv")

    # Standardize dates
    def _to_ymd(s):
        return pd.to_datetime(s, utc=True, errors="coerce").dt.strftime("%Y-%m-%d")
    df_ecmwf["date"] = _to_ymd(df_ecmwf["date"])
    df_reef["dates"] = _to_ymd(df_reef["dates"])
    df_reef = df_reef.rename(columns={"dates": "date"})
    for d in (df_slope, df_geom, df_ndvi):
        d["date"] = _to_ymd(d["date"])

    # Merge variables (outer) → remove exact dups → average duplicates
    df_merge = (
        df_ecmwf
        .merge(df_slope, on=["date", "transect_id"], how="outer")
        .merge(df_geom, on=["date", "transect_id"], how="outer")
        .merge(df_ndvi, on=["date", "transect_id"], how="outer")
        .merge(df_reef, on=["date", "transect_id"], how="outer")
    )
    df_merge = df_merge.drop_duplicates()
    df_merge = df_merge.groupby(["date", "transect_id"], as_index=False).mean(numeric_only=True)

    # Target
    df_target = df_target.drop(df_target.columns[0], axis=1).rename(columns={"dates": "date"})
    df_target_m = df_target.melt(id_vars=["date"], var_name="transect_id", value_name="shoreline_pos")
    df_target_m["date"] = _to_ymd(df_target_m["date"])

    # Inner-join with target, add monsoon flag
    df_final = pd.merge(df_merge, df_target_m, on=["date", "transect_id"], how="inner")
    df_final["date"] = pd.to_datetime(df_final["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df_final = add_monsoon_flag(df_final, start_month=monsoon_start, end_month=monsoon_end)

    # Basic cleaning
    df_final = df_final.dropna(subset=["shoreline_pos"])
    if "transect_id" in df_final.columns:
        df_final = df_final.drop(columns=["transect_id"])

    # Fill NaNs by yearly mean for selected columns (if present)
    df_final["date"] = pd.to_datetime(df_final["date"], errors="coerce")
    df_final["year"] = df_final["date"].dt.year
    for col in cols_to_fill:
        if col in df_final.columns:
            df_final[col] = df_final.groupby("year")[col].transform(lambda x: x.fillna(x.mean()))
    df_final = df_final.drop(columns=["year"])
    df_final["date"] = df_final["date"].dt.strftime("%Y-%m-%d")

    # Save the final file
    if output_path is None:
        output_path = f'./Model_Data_{sitename}/final_data_supplemented.csv'
    if save:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_final.to_csv(output_path, index=False)

    return df_final, output_path

def compile_island_data(site_options, output_path):
    dfs = []
    for site in site_options:
        file_path = f'./Model_Data_{site}/final_data_supplemented.csv'
        df = pd.read_csv(file_path)
        dfs.append(df)

    df_total = pd.concat(dfs, axis=0, ignore_index=True, sort=True)
    df_total = df_total[dfs[0].columns]  # ensure consistent column order
    df_total.to_csv(output_path, index=False)
    print(f"Compiled dataset saved to {output_path}")