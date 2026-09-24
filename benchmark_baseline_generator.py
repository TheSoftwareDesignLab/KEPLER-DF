import sys
import yaml
import shutil
import copy
from pathlib import Path

from src.main import main as kepler_main

def get_base_config():
    return {
        "simulation": {
            "num_scenarios": 5,
            "seed": 0,
            "semantic_enabled": False,
            "export_kml_visualization": True,
        },
        "payload": {
            "sensors_pool": ["OPT", "SAR"],
            "sensor_weights": [0.6, 0.4],
            "sensor_generation_rates": {"OPT": 80.0, "SAR": 40.00},
            "sensor_constraints": {
                "SAR": {"max_look_angle_deg": 40.0},
                "OPT": {"max_look_angle_deg": 30.0}
            },
            "storage_capacity_pool_mb": [128000.0, 512000.0],
            "storage_capacity_weights": [0.6, 0.7],
            "bands_config": {
                "X": {"weight": 0.6, "min_elevation_deg": 8.0, "max_slant_range_km": 2200.0, "downlink_rate_mb_s": 45.0},
                "S": {"weight": 0.2, "min_elevation_deg": 5.0, "max_slant_range_km": 2600.0, "downlink_rate_mb_s": 2.0},
            },
            "min_sensors_per_sat": 1,
            "max_sensors_per_sat": 2
        },
        "task_generation": {
            "polygon_ratio": 0.6,
            "min_area_deg": 0.05,
            "max_area_deg": 0.20,
            "min_duration": 5,
            "max_duration": 30,
            "min_release_delay": 0,
            "max_release_delay": 600,
            "min_lifetime": 1900,
            "max_lifetime": 86400,
            "priority_weights": [0.5, 0.3, 0.2]
        },
        "paths": {
            "gs_file_path": "data/ground_station.csv",
            "sat_file_path": "data/satellite_tle.csv"
        }
    }

def get_bounding_boxes(distribution_type: str):
    if distribution_type == "Global":
        return [{
            "name": "global_commercial_envelope",
            "lat_envelope": [-60.0, 60.0],
            "lon_envelope": [-180.0, 180.0]
        }]
    else:
        return [
            {"name": "andean_latam", "lat_envelope": [-5.0, 12.0], "lon_envelope": [-80.0, -65.0]},
            {"name": "western_europe", "lat_envelope": [35.0, 55.0], "lon_envelope": [-10.0, 20.0]},
            {"name": "southeast_asia", "lat_envelope": [-10.0, 15.0], "lon_envelope": [95.0, 130.0]},
            {"name": "north_america_conus", "lat_envelope": [25.0, 50.0], "lon_envelope": [-125.0, -70.0]}
        ]

def generate_baseline_benchmark_data():
    topologies = [
        {"name": "Walker Delta", "inc": 53.0, "alt": 550.0, "tag": "WD"},
        {"name": "Walker Star", "inc": 86.4, "alt": 780.0, "tag": "WS"},
        {"name": "Arbitrary", "tag": "ARB", "group": "weather"} 
    ]
    
    scales = [
        {"name": "Small", "sat_k": 8, "tasks_k": 150, "planes": 2},
        {"name": "Medium", "sat_k": 24, "tasks_k": 600, "planes": 4},
        {"name": "Large", "sat_k": 60, "tasks_k": 1500, "planes": 6}
    ]
    
    distributions = ["Clustered", "Global"]
    
    temp_yaml_path = "benchmark_temp_config.yaml"
    scenario_id = 1
    
    for topo in topologies:
        for scale in scales:
            for dist in distributions:
                dataset_name = f"DS_{scenario_id:02d}_{topo['tag']}_{scale['name']}_{dist}"
                print(f"\n{'='*60}\nPREPARING BENCHMARK SCENARIO: {dataset_name}\n{'='*60}")
                
                cfg = get_base_config()
                cfg["simulation"]["dataset_name"] = dataset_name
                cfg["simulation"]["constellation_type"] = topo["name"]
                cfg["simulation"]["sat_k"] = scale["sat_k"]
                cfg["simulation"]["gs_k"] = 3
                cfg["simulation"]["tasks_k"] = scale["tasks_k"]
                
                cfg["task_generation"]["bounding_boxes"] = get_bounding_boxes(dist)
                
                if topo["name"] in ["Walker Delta", "Walker Star"]:
                    cfg["simulation"]["walker_params"] = {
                        "t": scale["sat_k"],
                        "p": scale["planes"],
                        "f": 1,
                        "altitude_km": topo["alt"],
                        "inc": topo["inc"],
                        "base_id": 90000
                    }
                else:
                    cfg["simulation"]["sat_group_name"] = topo["group"]
                
                with open(temp_yaml_path, "w", encoding="utf-8") as f:
                    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
                
                sys.argv = ["main.py", temp_yaml_path]
                
                try:
                    kepler_main()
                except Exception as e:
                    print(f"[FATAL ERROR] Failed to generate scenario {dataset_name}: {e}")
                    continue
                
                output_dir = Path("data") / dataset_name
                if output_dir.exists():
                    shutil.copy(temp_yaml_path, output_dir / "execution_config.yaml")
                    print(f"[BENCHMARK] YAML successfully saved at: {output_dir / 'execution_config.yaml'}")
                
                scenario_id += 1

    if Path(temp_yaml_path).exists():
        Path(temp_yaml_path).unlink()
    
    print("\nBASELINE BENCHMARK DATA GENERATION COMPLETED.")

if __name__ == "__main__":
    generate_baseline_benchmark_data()