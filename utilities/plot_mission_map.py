import json
import pathlib
import urllib.request
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MapPolygon
from skyfield.api import EarthSatellite, load, wgs84

__all__ = ["generate_mission_world_map"]


def _load_json_data(file_path: str) -> dict:
    """
    Safely loads a JSON data file using standard Python library streams.
    """
    path = pathlib.Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Required data file registry missing at: {file_path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _fetch_geojson_from_url(url: str) -> dict:
    """
    Downloads official vector geometries from an open academic repository using 
    standard library streams to prevent Python 3.14 GIS compilation errors.
    """
    import ssl
    ssl_context = ssl._create_unverified_context()
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=15, context=ssl_context) as response:
            return json.loads(response.read().decode())
    except Exception as e:
        print(f"[WARN] Remote fetch failed for URL {url}: {e}. Skipping layer.")
        return {"type": "FeatureCollection", "features": []}


def _propagate_ground_track_skyfield(satellite: EarthSatellite, ts, start_dt: datetime, end_dt: datetime) -> tuple:
    """
    Propagates the satellite orbit step-by-step from AOS to LOS using Skyfield API.
    Bypasses manual geodetic matrix conversions by extracting sub-satellite positions natively.
    """
    lons, lats = [], []
    current_time = start_dt
    
    while current_time <= end_dt:
        t_skyfield = ts.utc(current_time.year, current_time.month, current_time.day,
                            current_time.hour, current_time.minute, current_time.second)
        
        geocentric = satellite.at(t_skyfield)
        subpoint = wgs84.subpoint(geocentric)
        
        lons.append(subpoint.longitude.degrees)
        lats.append(subpoint.latitude.degrees)
        
        current_time += timedelta(seconds=2)  
        
    return lons, lats


def generate_mission_world_map() -> None:
    """
    Compiles a dynamic, unified academic mission map.
    Plots exclusive capture windows within a strict 4-hour micro-window to prevent visual clutter,
    using dynamic matching colors between paths and target assets.
    """
    current_dir = pathlib.Path(__file__).parent.resolve()
    
    passes_json_path = current_dir.parent / "data" / "constellation_dataset_prueba" / "scenario_1" / "physics_passes_report.json"
    scenario_json_path = current_dir.parent / "data" / "constellation_dataset_prueba" / "scenario_1" / "scenario_report.json"
    output_dir = current_dir / "output"
    output_svg = output_dir / "satellite_mission_map.svg"
    
    passes_data = _load_json_data(str(passes_json_path))
    scenario_data = _load_json_data(str(scenario_json_path))
    
    target_passes = passes_data.get("target_passes", [])
    task_pool = scenario_data.get("targets", [])
    satellites_pool = scenario_data.get("satellites", [])

    if not target_passes:
        print("[WARN] No target imaging passes found to process.")
        return

    print("  - Initializing Skyfield Timescale Engine...")
    ts = load.timescale()

    selected_sat_id = int(target_passes[0]["satellite_id"])
    print(f"  - Dynamically selected active spacecraft from physical logs: NORAD {selected_sat_id}")

    sat_entry = next((s for s in satellites_pool if int(s["norad_id"]) == selected_sat_id), None)
    if not sat_entry:
        raise ValueError(f"Spacecraft NORAD {selected_sat_id} logs registered pases but TLE is missing in registry.")
    
    skyfield_satellite = EarthSatellite(sat_entry["tle_line1"], sat_entry["tle_line2"], 
                                        name=str(selected_sat_id), ts=ts)
    
    raw_passes_mapped = []
    for p in target_passes:
        if int(p["satellite_id"]) == selected_sat_id:
            try:
                aos_dt = datetime.strptime(p["aos_utc"], "%Y-%m-%d %H:%M:%S")
                los_dt = datetime.strptime(p["los_utc"], "%Y-%m-%d %H:%M:%S")
                raw_passes_mapped.append({"task_id": p["task_id"], "start": aos_dt, "end": los_dt})
            except (ValueError, KeyError):
                continue

    if not raw_passes_mapped:
        print(f"[WARN] No valid passes found for satellite {selected_sat_id}.")
        return

    raw_passes_mapped.sort(key=lambda x: x["start"])
    min_aos_global = raw_passes_mapped[0]["start"]
    
    max_micro_window_dt = min_aos_global + timedelta(hours=4)
    print(f"  - Restricting evaluation horizon to a strict 4-hour window: {min_aos_global} to {max_micro_window_dt}")

    sat_active_windows = [w for w in raw_passes_mapped if min_aos_global <= w["start"] <= max_micro_window_dt]

    regions_data = {}
    global_lats, global_lons = [], []
    for task in task_pool:
        r_tag = task.get("region_tag")
        if not r_tag:
            matched_pass = next((p for p in target_passes if p["task_id"] == task["task_id"]), None)
            r_tag = matched_pass.get("region_tag") if matched_pass else "Region_Alpha"
            
        coords = task.get("coordinates", [])
        if coords:
            if r_tag not in regions_data:
                regions_data[r_tag] = {"lats": [], "lons": []}
            for c in coords:
                lat_val, lon_val = float(c[0]), float(c[1])
                regions_data[r_tag]["lats"].append(lat_val)
                regions_data[r_tag]["lons"].append(lon_val)
                global_lats.append(lat_val)
                global_lons.append(lon_val)

    url_coastline = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_coastline.geojson"
    url_borders = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_admin_0_boundary_lines_land.geojson"

    print("  - Fetching academic vector coastline data for background map verification...")
    coastline_geojson = _fetch_geojson_from_url(url_coastline)
    
    print("  - Fetching international political borders land vectors...")
    borders_geojson = _fetch_geojson_from_url(url_borders)

    color_palette = ["#E6194B", "#3CB44B", "#FFE119", "#4363D8", "#F58231", "#911EB4", "#46F0F0", "#F032E6"]

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
    
    fig, ax = plt.subplots(figsize=(11, 8.5), dpi=300)

    lon_start, lon_end = min(global_lons) - 3.0, max(global_lons) + 3.0
    lat_start, lat_end = min(global_lats) - 3.0, max(global_lats) + 3.0

    ax.set_facecolor("#D4E6F1")  
    ax.set_xlim(lon_start, lon_end)
    ax.set_ylim(lat_start, lat_end)
    
    for feature in coastline_geojson.get("features", []):
        geom = feature.get("geometry", {})
        if geom.get("type") == "LineString":
            coords = geom.get("coordinates", [])
            ax.plot([c[0] for c in coords], [c[1] for c in coords], color="#4A5568", linewidth=1.2, zorder=2)
        elif geom.get("type") == "MultiLineString":
            for line in geom.get("coordinates", []):
                ax.plot([c[0] for c in line], [c[1] for c in line], color="#4A5568", linewidth=1.2, zorder=2)

    for feature in borders_geojson.get("features", []):
        geom = feature.get("geometry", {})
        if geom.get("type") == "LineString":
            coords = geom.get("coordinates", [])
            ax.plot([c[0] for c in coords], [c[1] for c in coords], color="#7F8C8D", linewidth=0.9, linestyle="--", zorder=3)
        elif geom.get("type") == "MultiLineString":
            for line in geom.get("coordinates", []):
                ax.plot([c[0] for c in line], [c[1] for c in line], color="#7F8C8D", linewidth=0.9, linestyle="--", zorder=3)

    for r_tag, r_coords in regions_data.items():
        r_lon_min, r_lon_max = min(r_coords["lons"]) - 0.3, max(r_coords["lons"]) + 0.3
        r_lat_min, r_lat_max = min(r_coords["lats"]) - 0.3, max(r_coords["lats"]) + 0.3
        
        bbox_vertices = [
            (r_lon_min, r_lat_min), (r_lon_max, r_lat_min),
            (r_lon_max, r_lat_max), (r_lon_min, r_lat_max), (r_lon_min, r_lat_min)
        ]
        config_poly = MapPolygon(bbox_vertices, facecolor="#34495E", edgecolor="#2C3E50", 
                                 linewidth=1.5, linestyle="-.", alpha=0.12, zorder=1)
        ax.add_patch(config_poly)
        
        ax.text(r_lon_min + 0.2, r_lat_max - 0.5, f"Region: {r_tag}", 
                fontsize=9, fontname="Times New Roman", fontstyle="italic", fontweight="bold", color="#1A252F", zorder=4)

    active_window_tasks = set(w["task_id"] for w in sat_active_windows)
    for task in task_pool:
        if task["task_id"] not in active_window_tasks:
            coords = task.get("coordinates", [])
            if not coords: continue
            if task.get("task_type") == "polygon":
                lons = [float(c[1]) for c in coords]
                lats = [float(c[0]) for c in coords]
                if lons[0] != lons[-1] or lats[0] != lats[-1]:
                    lons.append(lons[0])
                    lats.append(lats[0])
                ax.plot(lons, lats, color="#BDC3C7", linewidth=0.8, alpha=0.3, zorder=4)
                ax.fill(lons, lats, color="#BDC3C7", alpha=0.02, zorder=4)
            else:
                ax.scatter([float(coords[0][1])], [float(coords[0][0])], color="#BDC3C7", marker="d", s=35, alpha=0.3, zorder=4)

    for p_idx, window in enumerate(sat_active_windows):
        pair_color = color_palette[p_idx % len(color_palette)]
        task_id = window["task_id"]
        
        orbit_lons, orbit_lats = _propagate_ground_track_skyfield(skyfield_satellite, ts, window["start"], window["end"])
        
        if orbit_lons and orbit_lats:
            ax.plot(
                orbit_lons, orbit_lats,
                color=pair_color, linewidth=3.2, linestyle="-", zorder=12,
                label=f"Capture Pass: {task_id}"
            )
            
        matched_task = next((t for t in task_pool if t["task_id"] == task_id), None)
        if matched_task:
            t_coords = matched_task.get("coordinates", [])
            if not t_coords:
                continue
                
            if matched_task.get("task_type") == "polygon":
                lons = [float(c[1]) for c in t_coords]
                lats = [float(c[0]) for c in t_coords]
                if lons[0] != lons[-1] or lats[0] != lats[-1]:
                    lons.append(lons[0])
                    lats.append(lats[0])
                ax.plot(lons, lats, color=pair_color, linewidth=2.4, zorder=6)
                ax.fill(lons, lats, color=pair_color, alpha=0.35, zorder=5)
            else:
                ax.scatter(
                    [float(t_coords[0][1])], [float(t_coords[0][0])],
                    color=pair_color, marker="d", s=75, edgecolor="black", 
                    linewidth=0.9, zorder=6
                )
                
    ax.grid(True, linestyle="--", alpha=0.5, color="#B2BABB", zorder=1)
    ax.set_xlabel("Longitude (°E)", fontsize=10, fontname="Times New Roman")
    ax.set_ylabel("Latitude (°N)", fontsize=10, fontname="Times New Roman")

    ax.set_title(
        f"Unified Spatial Analysis Map - 4-Hour Micro-Window Intersections\nSkyfield API Dynamic Track Pairing for Spacecraft NORAD {selected_sat_id} & Active Target Matrix",
        fontsize=11, fontname="Times New Roman", fontweight="bold", pad=15
    )

    ax.legend(
        loc="lower left",
        prop={"family": "serif", "size": 9.2},
        facecolor="#FFFFFF",
        edgecolor="#000000",
        framealpha=0.95,
        frameon=True
    )

    plt.tight_layout()
    
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_svg), format="svg", bbox_inches="tight")
    plt.close(fig)
    
    print(f"[SUCCESS] High-fidelity paired micro-window map (4h horizon) safely generated and saved to: {output_svg.resolve()}")


if __name__ == "__main__":
    generate_mission_world_map()