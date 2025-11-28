import numpy as np
import pandas as pd
import os, json

def pick_baseline_idx(D, dates, baseline_year=2016):
    years = np.array([dt.year for dt in dates])
    nT, nK = D.shape
    baseline_idx = np.full(nT, -1, dtype=int)
    for ti in range(nT):
        idxs = np.where((years == baseline_year) & ~np.isnan(D[ti]))[0]
        if idxs.size == 0:  # Fallback: if that year is missing, use the earliest non-NaN across the whole period
            idxs = np.where(~np.isnan(D[ti]))[0]
        if idxs.size > 0:
            baseline_idx[ti] = idxs.min()
    return baseline_idx

def rmean_from_baseline(D_corr, baseline_idx, reducer=np.nanmedian):
    # Aggregate across transects using “corrected baseline distance” to obtain feature-scale R_mean
    base_vals = [D_corr[i, bi] if bi >= 0 else np.nan
                 for i, bi in enumerate(baseline_idx)]
    return float(reducer(np.array(base_vals, dtype=float)))

def make_deltas(D_corr, baseline_idx, R_mean=None):
    nT, nK = D_corr.shape
    Delta_d = np.full_like(D_corr, np.nan, dtype=float)
    for ti in range(nT):
        bi = baseline_idx[ti]
        if bi >= 0:
            Delta_d[ti] = D_corr[ti] - D_corr[ti, bi]
    Delta_d_p = None if (R_mean is None or R_mean <= 0) else (Delta_d / R_mean)
    return Delta_d, Delta_d_p

def save_reconstruction_meta(filepath, sitename,
                             t_cols, dates,
                             D_corr, baseline_idx,
                             R_mean, mu=None, sigma=None,
                             transects_geojson_path=None,
                             epsg=None, centroid=None,
                             baseline_year=2016, notes=None):

    baseline_d = []
    baseline_dates = []
    for i, bi in enumerate(baseline_idx):
        if bi >= 0 and not np.isnan(D_corr[i, bi]):
            baseline_d.append(float(D_corr[i, bi]))
            baseline_dates.append(pd.to_datetime(dates[bi]).isoformat())
        else:
            baseline_d.append(None)
            baseline_dates.append(None)

    meta = {
        "sitename": sitename,
        "baseline_year": int(baseline_year),
        "transects": list(t_cols),                       # Column order
        "baseline_index_per_transect": [int(bi) if bi >= 0 else None for bi in baseline_idx],
        "baseline_distance_per_transect_m": baseline_d,  # Required: add back during reconstruction
        "baseline_date_per_transect": baseline_dates,
        "R_mean_m": float(R_mean),                       # Required: Δd′ ↔ Δd
        "standardization": {
            "mu": None if mu is None else float(mu),
            "sigma": None if sigma is None else float(sigma)
        },
        "transects_geojson": transects_geojson_path,     # Recommended if reconstruction to XY is needed
        "epsg": epsg,
        "centroid": {"x": centroid[0], "y": centroid[1]} if centroid is not None else None,
        "notes": notes
    }

    out_json = os.path.join(filepath, f"{sitename}_reconstruction_metadata.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"Saved metadata: {out_json}")
