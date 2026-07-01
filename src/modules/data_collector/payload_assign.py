import random
from typing import List, Optional, Dict

__all__ = ["assign_satellite_payloads"]


def assign_satellite_payloads(
    satellites: List[any],
    available_sensors: List[str],
    available_bands: List[str],
    available_capacities: Optional[List[float]] = None,
    capacities_weights: Optional[List[float]] = None,
    sensor_weights: Optional[List[float]] = None,
    band_weights: Optional[List[float]] = None,
    sensor_generation_rates: Optional[Dict[str, float]] = None,
    band_downlink_rates: Optional[Dict[str, float]] = None,
    min_sensors_per_sat: int = 1,
    max_sensors_per_sat: int = 1,
    seed: Optional[int] = None
) -> List[any]:
    if not satellites:
        return []
        
    if not available_sensors or not available_bands:
        raise ValueError("Available sensors and bands configuration pools cannot be empty.")
        
    if min_sensors_per_sat < 1 or max_sensors_per_sat > len(available_sensors) or min_sensors_per_sat > max_sensors_per_sat:
        raise ValueError("Invalid sensor allocation bounds provided for payload assignment.")

    rng = random.Random(seed)
    
    if not available_capacities:
        available_capacities = [512000.0]
        capacities_weights = [1.0]

    rates_sensors = sensor_generation_rates or {}

    for sat in satellites:
        sat.band = rng.choices(available_bands, weights=band_weights, k=1)[0]
        if band_downlink_rates and sat.band in band_downlink_rates:
            sat.downlink_rate_mb_s = band_downlink_rates[sat.band]
        else:
            sat.downlink_rate_mb_s = 10.0

        sat.capacity_mb = rng.choices(available_capacities, weights=capacities_weights, k=1)[0]
        
        num_sensors = rng.randint(min_sensors_per_sat, max_sensors_per_sat)
        chosen_sensors = []
        current_sensors = list(available_sensors)
        current_weights = list(sensor_weights) if sensor_weights is not None else [1.0] * len(available_sensors)
        
        while len(chosen_sensors) < num_sensors:
            pick = rng.choices(current_sensors, weights=current_weights, k=1)[0]
            chosen_sensors.append(pick)
            idx = current_sensors.index(pick)
            current_sensors.pop(idx)
            current_weights.pop(idx)
            
        sat.sensors = chosen_sensors
        
        sat.sensor_generation_rates = {
            sensor: rates_sensors.get(sensor, 30.0) 
            for sensor in sat.sensors
        }
        
    return satellites