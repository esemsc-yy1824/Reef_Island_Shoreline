import os
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import matplotlib.pyplot as plt
import joblib
from shapely.geometry import LineString
from pyproj import Transformer
from matplotlib.patches import FancyArrow


def _nice_length(target):
    """Round target (meters) to a 'nice length' of 1/2/5×10^k"""
    if target <= 0 or not np.isfinite(target): return 0
    exp = int(np.floor(np.log10(target)))
    base = target / (10 ** exp)
    if base < 1.5: nice = 1
    elif base < 3.5: nice = 2
    elif base < 7.5: nice = 5
    else: nice = 10
    return nice * (10 ** exp)

def _add_scalebar(ax, bounds, units="m", frac=0.2, dy_frac=0.03, text=""):
    """Add a scalebar at the bottom-left corner (only for metric projections)"""
    xmin, xmax, ymin, ymax = bounds.left, bounds.right, bounds.bottom, bounds.top
    w = xmax - xmin; h = ymax - ymin
    target = w * frac
    L = _nice_length(target)
    if L <= 0: return
    x0 = xmin + w * 0.05
    y0 = ymin + h * 0.06
    ax.plot([x0, x0 + L], [y0, y0], color="black", lw=3, solid_capstyle="butt", zorder=10)
    ax.plot([x0, x0], [y0 - h*dy_frac, y0 + h*dy_frac], color="black", lw=2, zorder=10)
    ax.plot([x0 + L, x0 + L], [y0 - h*dy_frac, y0 + h*dy_frac], color="black", lw=2, zorder=10)
    label = f"{int(L)} {units}" if L < 1000 else f"{L/1000:.1f} km"
    ax.text(x0 + L/2, y0 + h*0.03, label, ha="center", va="bottom", fontsize=10, color="black", zorder=10)

def _add_north_arrow(ax, bounds):
    """Place a north arrow at the top-right corner (assuming north is up)"""
    xmin, xmax, ymin, ymax = bounds.left, bounds.right, bounds.bottom, bounds.top
    w = xmax - xmin; h = ymax - ymin
    x = xmax - w * 0.06
    y = ymax - h * 0.12
    ax.add_patch(FancyArrow(x, y, 0, h*0.06, width=w*0.005, head_width=w*0.02, head_length=h*0.03,
                            length_includes_head=True, color="black", zorder=10))
    ax.text(x, y + h*0.07, "N", ha="center", va="bottom", fontsize=11, color="black", zorder=10)

def quick_plot_using_tidal_truth(
    tidal_csv: str,
    transects_geojson: str,
    raster_path: str,
    meta_json: str,
    date_str: str = "2024-12-11",
    # —— Prediction inputs (priority: pred_eval_csv > y_pred+y_pred_transects > y_pred length match)
    pred_eval_csv: str | None = None,
    y_pred: np.ndarray | None = None,
    y_pred_transects: list[str] | None = None,
    y_scaler_path: str | None = None,
    # —— Plot options ——
    annotate: bool = False,
    point_size: int = 26,
    close_ring: bool = True,          # Whether to close the curve
    show_transect_names_every: int = 0,  # Label every N transects, 0 = none
    # —— Export options ——
    save_path: str | None = None,     # e.g. "figs/Keyodhoo_2024-12-11_pred-vs-true.png"; None = auto name
    save_dpi: int = 600,
):
    # ===== 0) Unified plot style (paper style) =====
    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })

    # ===== 1) Read tidal-corrected ground truth (meters) =====
    df = pd.read_csv(tidal_csv)
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed", case=False)]
    date_col = "date" if "date" in df.columns else "dates" if "dates" in df.columns else None
    if date_col is None:
        raise ValueError("Tidal-corrected CSV is missing 'date' or 'dates' column.")
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce", utc=True)
    target_day = pd.to_datetime(date_str).date()
    row = df.loc[df[date_col].dt.date == target_day].sort_values(date_col).head(1)
    if row.empty:
        raise ValueError(f"{date_str} has no record in {os.path.basename(tidal_csv)}.")
    t_cols_all = [c for c in df.columns if c.upper().startswith("T")]
    if not t_cols_all:
        raise ValueError("No T1..TN columns in tidal-corrected CSV.")
    t_cols_all = sorted(t_cols_all, key=lambda c: int("".join(filter(str.isdigit, c)) or 0))
    d_true_m_all = row.iloc[0][t_cols_all].to_numpy(dtype=float)  # Ground truth (meters)

    # ===== 2) Read meta (R_mean, baseline_d, transect order & EPSG) =====
    with open(meta_json, "r", encoding="utf-8") as f:
        meta = json.load(f)
    R_mean = float(meta["R_mean_m"])
    t_order_meta = [str(t) for t in meta["transects"]]
    baseline_d_map = {str(t): (float(b) if b is not None else np.nan)
                      for t, b in zip(meta["transects"], meta["baseline_distance_per_transect_m"])}

    # Final order = intersection of tidal CSV T columns & meta transects
    t_order = [t for t in t_cols_all if t in t_order_meta]
    if len(t_order) == 0:
        raise ValueError("No overlap between T columns in tidal CSV and meta['transects'].")
    idx_keep = [t_cols_all.index(t) for t in t_order]
    d_true_m = d_true_m_all[idx_keep]

    # ===== 3) Assemble 'transect-aligned Δd′ prediction' =====
    dp_aligned = None
    if pred_eval_csv is not None:
        dfp = pd.read_csv(pred_eval_csv)
        dfp = dfp.loc[:, ~dfp.columns.str.contains(r"^Unnamed", case=False)]
        if "date" not in dfp.columns: raise ValueError("pred_eval_csv missing 'date' column.")
        dfp["date"] = pd.to_datetime(dfp["date"], errors="coerce")
        sub = dfp[dfp["date"].dt.date == target_day]
        if sub.empty: raise ValueError(f"{date_str} has no record in pred_eval_csv.")
        if "transect" not in sub.columns: raise ValueError("pred_eval_csv must contain 'transect' column.")
        if "delta_dp_pred" in sub.columns:
            dp_map = {str(t): float(v) for t, v in zip(sub["transect"].astype(str), sub["delta_dp_pred"].astype(float))}
        elif ("z_pred" in sub.columns) and y_scaler_path:
            scaler = joblib.load(y_scaler_path)
            z = sub["z_pred"].to_numpy(dtype=float).reshape(-1,1)
            dp = scaler.inverse_transform(z).ravel()
            dp_map = {str(t): float(v) for t, v in zip(sub["transect"].astype(str), dp)}
        else:
            raise ValueError("pred_eval_csv has neither 'delta_dp_pred' nor reversible 'z_pred'.")
        dp_aligned = np.array([dp_map.get(t, np.nan) for t in t_order], dtype=float)

    elif y_pred is not None:
        dp = np.asarray(y_pred, dtype=float).ravel()
        if y_scaler_path:
            scaler = joblib.load(y_scaler_path)
            dp = scaler.inverse_transform(dp.reshape(-1,1)).ravel()
        if y_pred_transects is not None:
            name_map = {str(n): dp[i] for i, n in enumerate(y_pred_transects)}
            dp_aligned = np.array([name_map.get(t, np.nan) for t in t_order], dtype=float)
        else:
            if len(dp) != len(t_order):
                raise ValueError(f"y_pred length ({len(dp)}) ≠ number of transects ({len(t_order)}), and no y_pred_transects/pred_eval_csv provided.")
            dp_aligned = dp.copy()
    else:
        pass  # Only plot ground truth

    # ===== 4) Read transects & raster, prepare reprojection =====
    gdf_tr = gpd.read_file(transects_geojson)
    name_col = "name" if "name" in gdf_tr.columns else "transect" if "transect" in gdf_tr.columns else None
    if name_col is None:
        raise ValueError("GeoJSON has no 'name' or 'transect' column.")
    if gdf_tr.crs is None and meta.get("epsg"):
        try: gdf_tr.set_crs(meta["epsg"], inplace=True)
        except Exception: pass
    geom_map = {str(r[name_col]): r.geometry for _, r in gdf_tr.iterrows()}

    with rasterio.open(raster_path) as src:
        rgb = src.read([3,2,1]).astype("float32")
        bounds = src.bounds
        dst_crs = src.crs

    src_crs = gdf_tr.crs
    transformer = None
    if (src_crs is not None) and (dst_crs is not None) and (src_crs != dst_crs):
        transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)

    def _xy_on_line(line: LineString, dist_m: float):
        d = max(0.0, min(float(dist_m), line.length))
        p = line.interpolate(d, normalized=False)
        x, y = p.x, p.y
        if transformer: x, y = transformer.transform(x, y)
        return x, y

    # ===== 5) Compute 'meter-level prediction' and generate points =====
    d_pred_m = None
    if dp_aligned is not None:
        d_pred_m = np.array([
            (np.nan if np.isnan(baseline_d_map.get(t, np.nan))
             else baseline_d_map.get(t) + dp_aligned[i]*R_mean)
            for i, t in enumerate(t_order)
        ], dtype=float)

    true_xy, pred_xy, tr_lines_xy, tr_name_xy = [], [], [], []
    for i, t in enumerate(t_order):
        line = geom_map.get(t, None)
        if line is None or not isinstance(line, LineString):
            continue
        xs, ys = np.array(line.xy[0]), np.array(line.xy[1])
        if transformer: xs, ys = transformer.transform(xs, ys)
        tr_lines_xy.append((xs, ys))
        # Use midpoint of line as label position
        xm, ym = xs[len(xs)//2], ys[len(ys)//2]
        tr_name_xy.append((t, xm, ym))

        if not np.isnan(d_true_m[i]):
            true_xy.append(_xy_on_line(line, d_true_m[i]))
        if (d_pred_m is not None) and (not np.isnan(d_pred_m[i])):
            pred_xy.append(_xy_on_line(line, d_pred_m[i]))

    # Error metrics (meters)
    r2_txt, rmse_txt = "", ""
    if d_pred_m is not None and len(true_xy) == len(pred_xy) and len(true_xy) > 2:
        # Compare in meter-level distance (already aligned by t_order)
        m_true = np.array([d_true_m[i] for i in range(len(t_order)) if not np.isnan(d_true_m[i]) and (d_pred_m is not None and not np.isnan(d_pred_m[i]))])
        m_pred = np.array([d_pred_m[i] for i in range(len(t_order)) if not np.isnan(d_true_m[i]) and (d_pred_m is not None and not np.isnan(d_pred_m[i]))])
        if m_true.size > 1 and m_true.size == m_pred.size:
            ss_res = np.nansum((m_true - m_pred)**2)
            ss_tot = np.nansum((m_true - np.nanmean(m_true))**2)
            r2 = 1 - ss_res/ss_tot if ss_tot > 0 else np.nan
            rmse = np.sqrt(np.nanmean((m_true - m_pred)**2))
            r2_txt = f"R²={r2:.3f}"
            rmse_txt = f"RMSE={rmse:.2f} m"

    # ===== 6) Plot (paper style) =====
    rgb = np.transpose(np.clip(rgb / np.nanmax(rgb), 0, 1), (1,2,0))
    fig, ax = plt.subplots(figsize=(5.5, 5.5), dpi=150)  # Control physical size; use higher dpi when exporting
    ax.imshow(rgb, extent=[bounds.left, bounds.right, bounds.bottom, bounds.top], alpha=0.65)

    # 1) Transect frame: black for better visibility
    for xs, ys in tr_lines_xy:
        ax.plot(xs, ys, color="black", linewidth=0.9, alpha=0.9, zorder=2)

    # Optionally label some transects, avoid overcrowding
    if show_transect_names_every and show_transect_names_every > 0:
        for k, (t, x, y) in enumerate(tr_name_xy):
            if k % show_transect_names_every == 0:
                ax.text(x, y, t, fontsize=8, color="black",
                        ha="center", va="center", zorder=3,
                        bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.6))

    # 2) Ground truth and predictions (connect into ring for overall contour)
    def _poly(xs, ys, color, label):
        if len(xs) == 0: return
        if close_ring and len(xs) >= 2:
            xs = xs + [xs[0]]; ys = ys + [ys[0]]
        ax.plot(xs, ys, "-o", color=color, markersize=max(3.0, point_size**0.5),
                markeredgecolor="white", markeredgewidth=0.8, linewidth=1.3,
                label=label, alpha=0.95, zorder=4)

    xs_true = [x for x,_ in true_xy]; ys_true = [y for _,y in true_xy]
    xs_pred = [x for x,_ in pred_xy]; ys_pred = [y for _,y in pred_xy]
    _poly(xs_true, ys_true, color="#1E88E5", label="Truth (tidal-corrected, m)")
    if len(xs_pred) > 0:
        _poly(xs_pred, ys_pred, color="#D81B60", label="Predicted (m)")

    # 3) Scalebar + north arrow (more accurate in projected CRS)
    try:
        if dst_crs is not None and dst_crs.is_projected:
            _add_scalebar(ax, bounds, units="m", frac=0.2)
        _add_north_arrow(ax, bounds)
    except Exception:
        pass

    # 4) Put metrics in subtitle
    subtitle = f"{meta.get('sitename','')}_{date_str}"
    metrics_txt = " • ".join([s for s in [r2_txt, rmse_txt] if s])
    ax.set_title(f"{subtitle}" + (f"  ({metrics_txt})" if metrics_txt else ""), pad=8)

    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    ax.set_aspect('equal', adjustable='box')
    leg = ax.legend(loc="upper left", frameon=True, fancybox=True, framealpha=0.85)
    for lh in leg.legend_handles:
        try: lh.set_linewidth(1.5)
        except Exception: pass
    ax.grid(False)

    plt.tight_layout()

    # —— Save high-res figure (default auto-naming + dual format) ——
    if save_path is None:
        os.makedirs("./figs", exist_ok=True)
        safe_date = pd.to_datetime(date_str).strftime("%Y-%m-%d")
        sitename = str(meta.get("sitename", "island"))
        save_path = os.path.join("./figs", f"{sitename}_{safe_date}_pred-vs-true_tidalcorr")
    root, ext = os.path.splitext(save_path)
    if ext.lower() in (".png", ".pdf", ".svg"):
        out_base = root
        out_exts = [ext.lower()]
    else:
        out_base = save_path
        out_exts = [".png", ".pdf"]  # Export both raster+vector formats

    for e in out_exts:
        fname = out_base + e
        plt.savefig(fname, dpi=save_dpi, bbox_inches="tight")
        print(f"[saved] {fname}  (dpi={save_dpi})")

    plt.show()
