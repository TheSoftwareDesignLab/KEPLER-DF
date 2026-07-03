from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from skyfield.api import EarthSatellite, wgs84, load
from src.core.datatypes import SatelliteConfig, GroundStationConfig, TargetTask

__all__ = ["compute_infrastructure_passes", "compute_target_passes"]


def _get_timescale():
    """
    Retrieves the global Skyfield timescale reference instance.

    Returns:
        A Skyfield Timescale object initialized for ephemeris time translations.
    """
    return load.timescale()


def _calculate_lvlh_attitude(sat_object: EarthSatellite, t_skyfield, c_lat: float, c_lon: float) -> Tuple[float, float]:
    """
    Computes the localized LVLH pitch and roll angles for pointing at a geodetic target.

    Constructs a localized Local-Vertical Local-Horizontal (LVLH) reference frame tracking the 
    satellite platform's current state vector in space. Projects the slant-range line-of-sight 
    vector onto the coordinate axes to deduce pointing angles.

    Args:
        sat_object: Skyfield EarthSatellite instance mapping the tracking asset.
        t_skyfield: Skyfield Time instance defining the exact epoch of tracking evaluation.
        c_lat: Geodetic geocentric latitude coordinate of the target node.
        c_lon: Geodetic geocentric longitude coordinate of the target node.

    Returns:
        A tuple containing calculated (pitch_degrees, roll_degrees) platform orientation constraints.
    """
    geocentric = sat_object.at(t_skyfield)
    r_sat = geocentric.position.km
    v_sat = geocentric.velocity.km_per_s

    target_topo = wgs84.latlon(c_lat, c_lon, elevation_m=0.0)
    r_target = target_topo.at(t_skyfield).position.km

    rho_inercial = r_target - r_sat

    z_lvlh = -r_sat / np.linalg.norm(r_sat)
    h_orbit = np.cross(r_sat, v_sat)
    y_lvlh = -h_orbit / np.linalg.norm(h_orbit)
    x_lvlh = np.cross(y_lvlh, z_lvlh)

    R_inercial_to_lvlh = np.vstack([x_lvlh, y_lvlh, z_lvlh])
    rho_lvlh = R_inercial_to_lvlh.dot(rho_inercial)
    rx, ry, rz = rho_lvlh

    pitch_deg = float(np.degrees(np.arctan2(rx, -rz)))
    roll_deg = float(np.degrees(np.arctan2(ry, -rz)))

    return pitch_deg, roll_deg


def _refine_crossing(ts, sat, topo_target, t_before, t_after, target_el_deg, iterations=6):
    """
    Refines the exact crossing time of an elevation threshold using bisection.

    Args:
        ts: Skyfield Timescale reference instance.
        sat: Skyfield EarthSatellite platform under evaluation.
        topo_target: Skyfield GeographicPosition element defining the tracking site anchor.
        t_before: Skyfield Time step when the satellite's elevation sits below the threshold.
        t_after: Skyfield Time step when the satellite's elevation sits above the threshold.
        target_el_deg: Target elevation limit bound in degrees defining the boundary crossing.
        iterations: Total number of optimization cuts to execute for precision convergence.

    Returns:
        A refined Skyfield Time object capturing the instant of threshold intersection.
    """
    tb, ta = t_before, t_after
    for _ in range(iterations):
        tm = ts.tdb_jd((tb.tdb + ta.tdb) / 2.0)
        altb = (sat - topo_target).at(tb).altaz()[0].degrees
        altm = (sat - topo_target).at(tm).altaz()[0].degrees
        if (altb - target_el_deg) * (altm - target_el_deg) <= 0:
            ta = tm
        else:
            tb = tm
    return ts.tdb_jd((tb.tdb + ta.tdb) / 2.0)


def _calculate_geodetic_centroid_and_radius(task: TargetTask) -> Tuple[float, float, float]:
    """
    Extracts the geodetic center mass coordinate and tracking perimeter radius of a target task.

    Args:
        task: TargetTask dataclass object containing geodetic coordinates and envelope geometry data.

    Returns:
        A tuple mapping (centroid_latitude, centroid_longitude, maximum_radius_degrees).
    """
    if task.task_type == "point":
        return task.coordinates[0][0], task.coordinates[0][1], 0.0
    
    lats = [pt[0] for pt in task.coordinates[:-1]]
    lons = [pt[1] for pt in task.coordinates[:-1]]
    
    c_lat = float(np.mean(lats))
    c_lon = float(np.mean(lons))
    
    max_radius_deg = 0.0
    for lat, lon in task.coordinates[:-1]:
        distance = np.sqrt((lat - c_lat)**2 + (lon - c_lon)**2)
        if distance > max_radius_deg:
            max_radius_deg = float(distance)
            
    return c_lat, c_lon, max_radius_deg


def _vectorized_pass_finder(
    sat_object: EarthSatellite,
    topo_target,
    start_dt: datetime,
    end_dt: datetime,
    min_el_deg: float,
    c_lat: float,
    c_lon: float,
    step_seconds: int = 20
) -> List[Dict[str, Any]]:
    """
    Executes a vectorized discrete-step sweep to detect geometric visibility intervals.

    Evaluates the line-of-sight vectors across an array of time steps to isolate clusters
    where the target node climbs above an elevation constraint. Automatically executes numerical 
    bisection loops to discover highly precise Acquisition of Signal (AOS) and Loss of Signal (LOS) times.

    Args:
        sat_object: Pre-loaded Skyfield EarthSatellite tracking asset.
        topo_target: Skyfield GeographicPosition geodetic location model.
        start_dt: Chronological datetime lower boundary defining the sweep window.
        end_dt: Chronological datetime upper boundary defining the sweep window.
        min_el_deg: Minimum elevation mask cutoff boundary in degrees.
        c_lat: Center latitude of the tracking node.
        c_lon: Center longitude of the tracking node.
        step_seconds: Time step interval in seconds used to discretize the evaluation matrix.

    Returns:
        A list of unformatted dictionaries reporting raw pass properties and duration values.
    """
    ts = _get_timescale()
    total_seconds = (end_dt - start_dt).total_seconds()
    n_steps = int(total_seconds // step_seconds) + 1
    
    time_array = [start_dt + timedelta(seconds=i * step_seconds) for i in range(n_steps)]
    times = ts.utc(np.array(time_array))
    
    difference = sat_object - topo_target
    alt, az, distance = difference.at(times).altaz()
    alt_deg = alt.degrees
    
    above = alt_deg >= min_el_deg
    discovered_passes = []
    
    if not np.any(above):
        return discovered_passes

    idx = np.arange(len(above))
    starts = idx[(above) & np.hstack(([False], ~above[:-1]))]
    ends = idx[(above) & np.hstack((~above[1:], [False]))]

    for s, e in zip(starts, ends):
        if e <= s:
            continue

        s0 = max(s - 1, 0)
        e1 = min(e + 1, len(times) - 1)

        try:
            t_aos = _refine_crossing(ts, sat_object, topo_target, times[s0], times[s], min_el_deg)
            t_los = _refine_crossing(ts, sat_object, topo_target, times[e], times[e1], min_el_deg)
        except Exception:
            continue

        seg_alt = alt_deg[s:e + 1]
        if len(seg_alt) == 0 or np.all(np.isnan(seg_alt)):
            continue

        seg_idx_max = int(np.argmax(seg_alt))
        alt_max = float(seg_alt[seg_idx_max])
        t_max = times[s + seg_idx_max]

        range_aos = difference.at(t_aos).altaz()[2].km
        range_los = difference.at(t_los).altaz()[2].km
        
        aos_dt = t_aos.utc_datetime()
        los_dt = t_los.utc_datetime()
        duration_s = max(0, int((los_dt - aos_dt).total_seconds()))

        discovered_passes.append({
            "aos_dt": aos_dt,
            "tmax_dt": t_max.utc_datetime(),
            "los_dt": los_dt,
            "max_el_deg": alt_max,
            "range_aos_km": float(range_aos),
            "range_los_km": float(range_los),
            "duration_s": duration_s
        })

    return discovered_passes


def compute_infrastructure_passes(
    satellite: SatelliteConfig,
    ground_station: GroundStationConfig,
    start_dt_utc: datetime,
    end_dt_utc: datetime,
    bands_config: dict,
    step_seconds: int = 20
) -> List[Dict[str, Any]]:
    """
    Computes satellite downlink contact windows over a selected ground station asset.

    Initializes SGP4 structures from TLE inputs to discover tracking passes. Calibrates link limits 
    against communications frequency profiles, computes geometric orientation limits at peak elevation, 
    and determines transmission volumes based on hardware bit-rates.

    Args:
        satellite: Configuration model for the tracking satellite vehicle.
        ground_station: Geodetic database model mapping target tracking station coordinates.
        start_dt_utc: Start bound epoch defining the analytical horizon.
        end_dt_utc: End bound epoch defining the analytical horizon.
        bands_config: Configuration map matching telemetry tags to link elevation masks.
        step_seconds: Temporal resolution in seconds specifying step slicing granularity.

    Returns:
        A list of unified dictionaries reporting validated network downlink window properties.
    """
    ts = _get_timescale()
    sat_object = EarthSatellite(satellite.tle_line1, satellite.tle_line2, satellite.name)
    topo_target = wgs84.latlon(ground_station.latitude, ground_station.longitude, elevation_m=ground_station.elevation)
    
    band_info = bands_config.get(satellite.band, {}) if satellite.band else {}
    min_el_deg = float(band_info.get("min_elevation_deg", 10.0))
    
    raw_passes = _vectorized_pass_finder(
        sat_object, topo_target, start_dt_utc, end_dt_utc, min_el_deg,
        ground_station.latitude, ground_station.longitude, step_seconds
    )
    
    formatted_passes = []
    tx_rate = satellite.downlink_rate_mb_s if satellite.downlink_rate_mb_s is not None else 10.0

    for p in raw_passes:
        max_downlink_capacity_mb = float(p["duration_s"] * tx_rate)
        t_max_skyfield = ts.from_datetime(p["tmax_dt"].replace(tzinfo=timezone.utc))
        pitch_max, roll_max = _calculate_lvlh_attitude(sat_object, t_max_skyfield, ground_station.latitude, ground_station.longitude)

        formatted_passes.append({
            "satellite_id": satellite.norad_id,
            "ground_station_id": ground_station.id,
            "aos_utc": p["aos_dt"].strftime("%Y-%m-%d %H:%M:%S"),
            "tmax_utc": p["tmax_dt"].strftime("%Y-%m-%d %H:%M:%S"),
            "los_utc": p["los_dt"].strftime("%Y-%m-%d %H:%M:%S"),
            "max_el_deg": p["max_el_deg"],
            "range_aos_km": p["range_aos_km"],
            "range_los_km": p["range_los_km"],
            "duration_s": p["duration_s"],
            "lvlh_target_pitch_deg": pitch_max,
            "lvlh_target_roll_deg": roll_max,
            "estimated_transmission_capacity_mb": max_downlink_capacity_mb
        })
    return formatted_passes


def compute_target_passes(
    satellite: SatelliteConfig,
    task: TargetTask,
    simulation_start_utc: datetime,
    min_el_deg: float,
    step_seconds: int = 20,
    min_duration: int = 5,
    max_duration: int = 30
) -> List[Dict[str, Any]]:
    """
    Calculates operational imaging access passes over a specified mission target.

    Propagates the platform orbital vector strictly bounded between individual task availability 
    release times and expiration deadlines. Computes swath transit time scales for polygonal targets 
    or stochastic snapshot durations for discrete point coordinates, outputting estimated bit-stream 
    data generation volumes and coordinate limits for subsequent optimization phases.

    Args:
        satellite: Configuration model for the candidate space platform.
        task: TargetTask instance defining operational targets and timeline limits.
        simulation_start_utc: Core reference epoch tracking the absolute t0 baseline.
        min_el_deg: Minimum elevation mask cutoff value in degrees for the optical sensor.
        step_seconds: Temporal step parameter in seconds for discrete sweep slicing.
        min_duration: Absolute lower threshold for point-target scanning intervals.
        max_duration: Absolute upper capping limit defining max continuous sensor activations.

    Returns:
        A list of fully resolved target pass dictionaries detailing imaging feasibility.
    """
    ts = _get_timescale()
    sat_object = EarthSatellite(satellite.tle_line1, satellite.tle_line2, satellite.name)
    
    c_lat, c_lon, task_radius_deg = _calculate_geodetic_centroid_and_radius(task)
    topo_target = wgs84.latlon(c_lat, c_lon, elevation_m=0.0)
    
    task_release_dt = simulation_start_utc + timedelta(seconds=task.release_time)
    task_deadline_dt = simulation_start_utc + timedelta(seconds=task.deadline)
    
    raw_passes = _vectorized_pass_finder(sat_object, topo_target, task_release_dt, task_deadline_dt, min_el_deg, c_lat, c_lon, step_seconds)
    
    mean_motion_rad_min = sat_object.model.no
    satellite_speed_deg_s = (mean_motion_rad_min * (180.0 / np.pi)) / 60.0
    
    temporal_buffer = 0.0
    if task.task_type == "polygon" and satellite_speed_deg_s > 0:
        temporal_buffer = task_radius_deg / satellite_speed_deg_s
        
    valid_passes = []
    primary_sensor = task.required_sensors[0] if task.required_sensors else "VIS"
    rates_map = satellite.sensor_generation_rates if satellite.sensor_generation_rates is not None else {}
    data_ingestion_rate = rates_map.get(primary_sensor, 30.0)

    for p in raw_passes:
        adjusted_aos = p["aos_dt"] - timedelta(seconds=temporal_buffer)
        adjusted_los = p["los_dt"] + timedelta(seconds=temporal_buffer)
        
        clipped_aos = max(adjusted_aos, task_release_dt)
        clipped_los = min(adjusted_los, task_deadline_dt)
        
        if clipped_los <= clipped_aos:
            continue
            
        visibility_duration_s = int((clipped_los - clipped_aos).total_seconds())
        
        if task.task_type == "polygon" and satellite_speed_deg_s > 0:
            polygon_transit_time_s = (2.0 * task_radius_deg) / satellite_speed_deg_s
            sensor_active_duration_s = min(int(polygon_transit_time_s), visibility_duration_s)
            
            mid_point_dt = clipped_aos + timedelta(seconds=visibility_duration_s / 2.0)
            img_start_dt = mid_point_dt - timedelta(seconds=sensor_active_duration_s / 2.0)
            img_end_dt = mid_point_dt + timedelta(seconds=sensor_active_duration_s / 2.0)
        else:
            seed_hash = int(satellite.norad_id) + hash(task.task_id) + int(p["aos_dt"].timestamp())
            rng = np.random.default_rng(abs(seed_hash) % (2**32))
            
            random_duration = rng.integers(int(min_duration), int(max_duration) + 1)
            sensor_active_duration_s = min(int(random_duration), visibility_duration_s)
            
            img_start_dt = p["tmax_dt"] - timedelta(seconds=sensor_active_duration_s / 2.0)
            img_end_dt = p["tmax_dt"] + timedelta(seconds=sensor_active_duration_s / 2.0)

        img_start_dt = max(img_start_dt, clipped_aos)
        img_end_dt = min(img_end_dt, clipped_los)
        sensor_active_duration_s = max(0, int((img_end_dt - img_start_dt).total_seconds()))
        
        generated_data_volume_mb = float(sensor_active_duration_s * data_ingestion_rate)
        
        t_start_skyfield = ts.from_datetime(img_start_dt.replace(tzinfo=timezone.utc))
        t_end_skyfield = ts.from_datetime(img_end_dt.replace(tzinfo=timezone.utc))
        
        pitch_start, roll_start = _calculate_lvlh_attitude(sat_object, t_start_skyfield, c_lat, c_lon)
        pitch_end, roll_end = _calculate_lvlh_attitude(sat_object, t_end_skyfield, c_lat, c_lon)
            
        valid_passes.append({
            "satellite_id": satellite.norad_id,
            "task_id": task.task_id,
            "region_tag": task.region_tag,
            "aos_utc": clipped_aos.strftime("%Y-%m-%d %H:%M:%S"),
            "los_utc": clipped_los.strftime("%Y-%m-%d %H:%M:%S"),
            "max_el_deg": p["max_el_deg"],
            "visibility_duration_s": visibility_duration_s,
            "imaging_start_utc": img_start_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "imaging_end_utc": img_end_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "sensor_imaging_duration_s": sensor_active_duration_s,
            "lvlh_start_pitch_deg": pitch_start,
            "lvlh_start_roll_deg": roll_start,
            "lvlh_end_pitch_deg": pitch_end,
            "lvlh_end_roll_deg": roll_end,
            "estimated_onboard_data_generation_mb": generated_data_volume_mb,
            "is_feasible": True
        })
        
    return valid_passes