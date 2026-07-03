import json
import pathlib
from datetime import datetime, timezone
from typing import Optional
from src.core.datatypes import CollectedContext
from .pass_calculator import compute_target_passes, compute_infrastructure_passes

__all__ = ["physics_engine_main"]


def _export_physics_report(all_infra_passes: list, all_target_passes: list, context: CollectedContext, simulation_start_utc: datetime, simulation_end_utc: datetime, output_path: str) -> None:
    """
    Exports a structured JSON compilation detailing computed line-of-sight access intervals.

    Args:
        all_infra_passes: List of compiled satellite-to-ground-station visibility data structures.
        all_target_passes: List of compiled satellite-to-geographical-target access windows.
        context: CollectedContext instance aggregating the constellation hardware and target registries.
        simulation_start_utc: Anchor datetime specifying the lower boundary of the evaluation horizon.
        simulation_end_utc: Anchor datetime specifying the upper boundary of the evaluation horizon.
        output_path: System string path location defining where the JSON file report container will be written.
    """
    path = pathlib.Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    report_data = {
        "metadata": {
            "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            "simulation_start_utc": simulation_start_utc.strftime("%Y-%m-%d %H:%M:%S"),
            "simulation_end_utc": simulation_end_utc.strftime("%Y-%m-%d %H:%M:%S"),
            "total_satellites": len(context.satellites),
            "total_ground_stations": len(context.ground_stations),
            "total_targets": len(context.targets),
            "compiled_infrastructure_passes": len(all_infra_passes),
            "compiled_target_passes": len(all_target_passes)
        },
        "infrastructure_passes": all_infra_passes,
        "target_passes": all_target_passes
    }
    
    with path.open("w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)


def physics_engine_main(
    context: CollectedContext,
    bands_config: dict,
    simulation_start_utc: datetime,
    simulation_end_utc: datetime,
    output_path: str = "data/physics_passes_report.json",
    step_seconds: int = 20,
    min_duration: int = 5,
    max_duration: int = 30
) -> None:
    """
    Orchestrates the orbital geometry simulation phase to discover geometric visibility intervals.

    Evaluates the complete Cartesian product of the constellation fleet against down-selected 
    ground tracking stations to discover down-link visibility windows. Concurrently propagates orbits 
    over point and polygonal target envelopes to chart payload line-of-sight access passes, updating 
    dynamic task coordinate alignments against the final localized LVLH pitch and roll geometric constraints.

    Args:
        context: CollectedContext aggregate mapping active space platforms, tracking hardware, and task geometries.
        bands_config: Mapping reference defining hardware frequency properties and link budget constraints.
        simulation_start_utc: Upper chronological anchor defining the start of the SGP4 propagation timeline.
        simulation_end_utc: Lower chronological anchor defining the termination of the SGP4 propagation timeline.
        output_path: System string path target defining where the compiled orbital visibility matrices are saved.
        step_seconds: Temporal integration step size in seconds specifying the geometric sampling granularity.
        min_duration: Minimum required window length in seconds for a target visibility pass to be considered viable.
        max_duration: Maximum allowed capping limit in seconds for a single continuous target observation task.
    """
    all_infra_passes = []
    for gs in context.ground_stations:
        for sat in context.satellites:
            passes = compute_infrastructure_passes(
                satellite=sat,
                ground_station=gs,
                start_dt_utc=simulation_start_utc,
                end_dt_utc=simulation_end_utc,
                bands_config=bands_config,
                step_seconds=step_seconds
            )
            all_infra_passes.extend(passes)

    all_target_passes = []
    for task in context.targets:
        for sat in context.satellites:
            passes = compute_target_passes(
                satellite=sat,
                task=task,
                simulation_start_utc=simulation_start_utc,
                min_el_deg=10.0,
                step_seconds=step_seconds,
                min_duration=min_duration,
                max_duration=max_duration
            )
            all_target_passes.extend(passes)

            if passes:
                task.assigned_lvlh_pitch_deg = passes[-1]["lvlh_end_pitch_deg"]
                task.assigned_lvlh_roll_deg = passes[-1]["lvlh_end_roll_deg"]

    _export_physics_report(
        all_infra_passes=all_infra_passes,
        all_target_passes=all_target_passes,
        context=context,
        simulation_start_utc=simulation_start_utc,
        simulation_end_utc=simulation_end_utc,
        output_path=output_path
    )