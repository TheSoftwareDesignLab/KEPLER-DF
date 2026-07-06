import json
import pathlib
import urllib.request
import ssl
from datetime import datetime, timezone, timedelta
import matplotlib.pyplot as plt
from skyfield.api import EarthSatellite, load, wgs84
import numpy as np

__all__ = ["generate_infrastructure_world_map"]


def _resolve_path(path_str: str) -> pathlib.Path:
    p = pathlib.Path(path_str)
    if p.is_absolute():
        return p
    if p.exists():
        return p.resolve()
    return (pathlib.Path(__file__).resolve().parent.parent / path_str).resolve()


def _load_json_data(file_path: pathlib.Path) -> dict:
    if not file_path.exists():
        raise FileNotFoundError(f"Required data file missing at: {file_path}")
    with file_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _fetch_geojson_from_url(url: str) -> dict:
    ssl_context = ssl._create_unverified_context()
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=15, context=ssl_context) as response:
            return json.loads(response.read().decode())
    except Exception as e:
        print(f"[WARN] Failed to fetch remote geojson boundaries from {url}: {e}. Skipping background layer.")
        return {"type": "FeatureCollection", "features": []}


def _get_satellite_ground_track(tle_line1: str, tle_line2: str, name: str, start_dt: datetime, end_dt: datetime, step_seconds: int = 30) -> list:
    ts = load.timescale()
    sat_object = EarthSatellite(tle_line1, tle_line2, name)
    
    start_naive = start_dt.astimezone(timezone.utc).replace(tzinfo=None)
    end_naive = end_dt.astimezone(timezone.utc).replace(tzinfo=None)
    
    total_seconds = int((end_naive - start_naive).total_seconds())
    n_steps = max(2, total_seconds // step_seconds)
    
    time_array = [start_naive + timedelta(seconds=i * step_seconds) for i in range(n_steps)]
    times = ts.utc(
        np.array([t.year for t in time_array]),
        np.array([t.month for t in time_array]),
        np.array([t.day for t in time_array]),
        np.array([t.hour for t in time_array]),
        np.array([t.minute for t in time_array]),
        np.array([t.second for t in time_array])
    )
    
    geocentric = sat_object.at(times)
    subpoints = wgs84.subpoint(geocentric)
    
    lats = subpoints.latitude.degrees
    lons = subpoints.longitude.degrees
    
    segments = []
    curr_segment_lons = []
    curr_segment_lats = []
    
    for lat, lon in zip(lats, lons):
        if curr_segment_lons and abs(lon - curr_segment_lons[-1]) > 180.0:
            segments.append((curr_segment_lons, curr_segment_lats))
            curr_segment_lons = [lon]
            curr_segment_lats = [lat]
        else:
            curr_segment_lons.append(lon)
            curr_segment_lats.append(lat)
            
    if curr_segment_lons:
        segments.append((curr_segment_lons, curr_segment_lats))
        
    return segments


def generate_infrastructure_world_map(
    scenario_report_path: str = "data/constellation_dataset_prueba/scenario_2/scenario_report.json",
    physics_report_path: str = "data/constellation_dataset_prueba/scenario_2/physics_passes_report.json",
    output_image_path: str = "data/constellation_dataset_prueba/scenario_2/infrastructure_downlink_map.png",
    selected_satellite_id: Optional[int] = None
) -> None:
    scenario_path = _resolve_path(scenario_report_path)
    physics_path = _resolve_path(physics_report_path)
    output_path = _resolve_path(output_image_path)
    
    scenario_data = _load_json_data(scenario_path)
    physics_data = _load_json_data(physics_path)
    
    metadata = scenario_data.get("metadata", {})
    t0_str = metadata.get("anchored_tle_epoch_utc")
    if t0_str:
        t0 = datetime.strptime(t0_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    else:
        t0 = datetime.now(timezone.utc)
        
    satellites = scenario_data.get("satellites", [])
    ground_stations = scenario_data.get("ground_stations", [])
    infra_passes = physics_data.get("infrastructure_passes", [])
    
    if not ground_stations:
        print("[WARN] No ground stations found in the report. Map generation skipped.")
        return

    if not selected_satellite_id and satellites:
        sat_pass_counts = {}
        for p_pass in infra_passes:
            sat_id = p_pass.get("satellite_id")
            if sat_id:
                sat_pass_counts[sat_id] = sat_pass_counts.get(sat_id, 0) + 1
        if sat_pass_counts:
            selected_satellite_id = max(sat_pass_counts, key=sat_pass_counts.get)
        else:
            selected_satellite_id = satellites[0]["norad_id"]

    satellites = [s for s in satellites if s["norad_id"] == selected_satellite_id]
    if not satellites:
        print(f"[WARN] Selected satellite ID {selected_satellite_id} not found in scenario report.")
        return

    num_stations = len(ground_stations)
    if num_stations == 1:
        cols = 1
        rows = 1
        fig, axes = plt.subplots(1, 1, figsize=(10, 8), dpi=150)
        axes = [axes]
    else:
        cols = 2
        rows = (num_stations + 1) // 2
        fig, axes = plt.subplots(rows, cols, figsize=(14, 5.5 * rows), dpi=150)
        axes = axes.flatten()

    countries_url = "https://raw.githubusercontent.com/datasets/geo-boundaries-world-110m/master/countries.geojson"
    world_geojson = _fetch_geojson_from_url(countries_url)
    
    sat_tle_map = {}
    for sat in satellites:
        sat_tle_map[sat["norad_id"]] = {
            "tle_line1": sat["tle_line1"],
            "tle_line2": sat["tle_line2"],
            "name": sat["name"]
        }

    sat_colors = ["#8E44AD", "#2E86C1", "#E74C3C", "#27AE60", "#F39C12", "#16A085"]
    sat_color_map = {sat["norad_id"]: sat_colors[i % len(sat_colors)] for i, sat in enumerate(satellites)}

    for idx, gs in enumerate(ground_stations):
        ax = axes[idx]
        gs_id = gs["id"]
        gs_name = gs["name"]
        gs_lat = gs["latitude"]
        gs_lon = gs["longitude"]
        
        if world_geojson and "features" in world_geojson:
            for feature in world_geojson["features"]:
                geom = feature["geometry"]
                if geom["type"] == "Polygon":
                    coords_list = [geom["coordinates"]]
                elif geom["type"] == "MultiPolygon":
                    coords_list = geom["coordinates"]
                else:
                    continue
                    
                for poly_coords in coords_list:
                    for ring in poly_coords:
                        lons_p = [pt[0] for pt in ring]
                        lats_p = [pt[1] for pt in ring]
                        ax.plot(lons_p, lats_p, color="#D5D8DC", linewidth=0.6, zorder=1)
                        ax.fill(lons_p, lats_p, color="#F4F6F7", zorder=0)

        ax.scatter(gs_lon, gs_lat, color="#E74C3C", marker="^", s=130, edgecolor="#2C3E50", linewidth=1.5, label=f"GS: {gs_name}", zorder=7)

        processed_passes = set()
        processed_orbits = set()
        for p_pass in infra_passes:
            if p_pass.get("ground_station_id") == gs_id:
                sat_id = p_pass["satellite_id"]
                if sat_id != selected_satellite_id:
                    continue
                
                color = sat_color_map.get(sat_id, "#8E44AD")
                sat_info = sat_tle_map.get(sat_id)
                if not sat_info:
                    continue
                
                aos_str = p_pass.get("aos_utc")
                los_str = p_pass.get("los_utc")
                
                if aos_str and los_str:
                    aos_dt = datetime.strptime(aos_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
                    los_dt = datetime.strptime(los_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
                    
                    orbit_start = aos_dt - timedelta(minutes=45)
                    orbit_end = los_dt + timedelta(minutes=45)
                    
                    orbit_key = f"{sat_id}_{int(orbit_start.timestamp() // 600)}"
                    if orbit_key not in processed_orbits:
                        orbit_segments = _get_satellite_ground_track(
                            sat_info["tle_line1"],
                            sat_info["tle_line2"],
                            sat_info["name"],
                            orbit_start,
                            orbit_end,
                            step_seconds=20
                        )
                        for seg_lons, seg_lats in orbit_segments:
                            ax.plot(seg_lons, seg_lats, color=color, linestyle="--", linewidth=0.8, alpha=0.4, zorder=2)
                        processed_orbits.add(orbit_key)
                    
                    pass_segments = _get_satellite_ground_track(
                        sat_info["tle_line1"],
                        sat_info["tle_line2"],
                        sat_info["name"],
                        aos_dt,
                        los_dt,
                        step_seconds=1
                    )
                    
                    label_str = f"Downlink Sat {sat_id}"
                    for seg_lons, seg_lats in pass_segments:
                        if label_str not in processed_passes:
                            ax.plot(seg_lons, seg_lats, color=color, linewidth=2.5, linestyle="-", label=label_str, zorder=6)
                            processed_passes.add(label_str)
                        else:
                            ax.plot(seg_lons, seg_lats, color=color, linewidth=2.5, linestyle="-", zorder=6)

        ax.set_xlim(gs_lon - 20, gs_lon + 20)
        ax.set_ylim(gs_lat - 15, gs_lat + 15)
        ax.grid(True, linestyle=":", alpha=0.6, color="#BDC3C7", zorder=1)
        
        ax.set_xlabel("Longitude (°E)", fontsize=9, fontname="DejaVu Sans")
        ax.set_ylabel("Latitude (°N)", fontsize=9, fontname="DejaVu Sans")
        
        ax.set_title(f"Station: {gs_name}", fontsize=11, fontweight="bold", pad=10, fontname="DejaVu Sans")
        
        ax.legend(
            loc="upper right",
            prop={"size": 7.5, "family": "DejaVu Sans"},
            facecolor="#FFFFFF",
            edgecolor="#BDC3C7",
            framealpha=0.9,
            frameon=True
        )

    for extra_idx in range(num_stations, len(axes)):
        fig.delaxes(axes[extra_idx])

    fig.suptitle(
        f"Multi-Station Infrastructure Analysis - Downlink Windows for Sat {selected_satellite_id}\nSGP4 Physical Orbit Propagation & Active Ground Station Intersections",
        fontsize=13, fontweight="bold", y=0.98, fontname="DejaVu Sans"
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    
    print(f"\n[SUCCESS] Infrastructure downlinks map saved successfully at: {output_path.resolve()}")


if __name__ == "__main__":
    print("==========================================================")
    print(" Launching Kepler Downlink Visualizer...")
    print("==========================================================")
    
    SCENARIO_REPORT = "data/constellation_dataset_prueba/scenario_1/scenario_report.json"
    PHYSICS_REPORT = "data/constellation_dataset_prueba/scenario_1/physics_passes_report.json"
    OUTPUT_MAP = "utilities/output/satellite_downlink_map.png"

    try:
        generate_infrastructure_world_map(
            scenario_report_path=SCENARIO_REPORT,
            physics_report_path=PHYSICS_REPORT,
            output_image_path=OUTPUT_MAP
        )
    except FileNotFoundError as fnf:
        print(f"\n[ERROR] Required report file missing: {fnf}")
        print("-> Make sure you run 'main.py' first to generate report files inside the 'data/' directory.")
    except Exception as e:
        print(f"\n[ERROR] An unexpected error occurred during map rendering: {e}")