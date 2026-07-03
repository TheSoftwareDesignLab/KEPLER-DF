import json
import pathlib
from datetime import datetime, timezone, timedelta
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

__all__ = ["generate_academic_gantt"]


def _load_physics_passes(json_path: str) -> list:
    """
    Loads infrastructure contact passes from a compiled physics engine JSON report.
    """
    path = pathlib.Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"The specified physics pass report file does not exist at: {json_path}")
        
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("infrastructure_passes", [])


def generate_academic_gantt() -> None:
    """
    Processes orbital contact intervals and exports a faceted, high-contrast academic Gantt chart
    vector SVG.
    """
    current_dir = pathlib.Path(__file__).parent.resolve()
    
    input_json = current_dir.parent / "data" / "constellation_dataset_prueba" / "scenario_1" / "physics_passes_report.json"
    output_dir = current_dir / "output"
    output_svg = output_dir / "satellite_passes_gantt.svg"
    
    raw_passes = _load_physics_passes(str(input_json))
    
    if not raw_passes:
        print(f"[WARN] No infrastructure contact windows discovered inside registry: {input_json}")
        return

    parsed_passes = []
    for passport in raw_passes:
        try:
            aos_dt = datetime.strptime(passport["aos_utc"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
            los_dt = datetime.strptime(passport["los_utc"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        except ValueError:
            continue

        parsed_passes.append({
            "Satellite": f"NORAD {passport['satellite_id']}",
            "Station": passport["ground_station_id"].replace("_", " ").title(),
            "AOS": aos_dt,
            "LOS": los_dt,
            "Capacity": float(passport["estimated_transmission_capacity_mb"]),
            "Duration": int(passport["duration_s"])
        })

    if not parsed_passes:
        print("[WARN] No valid timestamps parsed from data source.")
        return

    parsed_passes.sort(key=lambda x: x["AOS"])
    min_aos = parsed_passes[0]["AOS"]
    max_window_dt = min_aos + timedelta(hours=12)

    filtered_passes = [
        p for p in parsed_passes 
        if min_aos <= p["AOS"] <= max_window_dt
    ]

    if not filtered_passes:
        print("[WARN] Filtered 12-hour time window contains zero data points.")
        return

    stations = sorted(list(set(p["Station"] for p in filtered_passes)))
    satellites = sorted(list(set(p["Satellite"] for p in filtered_passes)), reverse=True)
    sat_y_map = {sat: idx for idx, sat in enumerate(satellites)}

    fig = make_subplots(
        rows=len(stations), 
        cols=1, 
        shared_xaxes=True,
        subplot_titles=stations,
        vertical_spacing=0.08
    )

    all_capacities = [p["Capacity"] for p in filtered_passes]
    min_cap, max_cap = min(all_capacities), max(all_capacities)
    cap_range = (max_cap - min_cap) if (max_cap - min_cap) > 0 else 1.0

    cividis_scale = [
        [0.0, "rgb(0,0,51)"],
        [0.3, "rgb(51,51,120)"],
        [0.6, "rgb(120,120,120)"],
        [1.0, "rgb(255,255,153)"]
    ]

    def _sample_cividis(val: float) -> str:
        norm = (val - min_cap) / cap_range
        norm = max(0.0, min(1.0, norm))
        for i in range(len(cividis_scale) - 1):
            if cividis_scale[i][0] <= norm <= cividis_scale[i+1][0]:
                t = (norm - cividis_scale[i][0]) / (cividis_scale[i+1][0] - cividis_scale[i][0])
                c1 = [int(x) for x in cividis_scale[i][1][4:-1].split(",")]
                c2 = [int(x) for x in cividis_scale[i+1][1][4:-1].split(",")]
                rgb = [int(c1[j] + t * (c2[j] - c1[j])) for j in range(3)]
                return f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"
        return cividis_scale[-1][1]

    for pass_idx, p in enumerate(filtered_passes):
        st_idx = stations.index(p["Station"]) + 1
        y_val = sat_y_map[p["Satellite"]]
        color_str = _sample_cividis(p["Capacity"])

        fig.add_trace(
            go.Scatter(
                x=[p["AOS"], p["LOS"]],
                y=[y_val, y_val],
                mode="lines",
                line=dict(color=color_str, width=14),
                showlegend=False,
                hoverinfo="text",
                hovertext=(
                    f"<b>{p['Satellite']}</b><br>"
                    f"Station: {p['Station']}<br>"
                    f"Capacity: {p['Capacity']:.2f} MB<br>"
                    f"Duration: {p['Duration']}s"
                )
            ),
            row=st_idx,
            col=1
        )

    fig.add_trace(
        go.Scatter(
            x=[None], y=[None],
            mode="markers",
            marker=dict(
                colorscale="Cividis",
                cmin=min_cap,
                cmax=max_cap,
                showscale=True,
                colorbar=dict(
                    title=dict(text="Capacity (MB)", font=dict(family="Serif", size=10)),
                    tickfont=dict(family="Serif", size=9),
                    thickness=12,
                    len=0.7,
                    y=0.5
                )
            ),
            showlegend=False
        )
    )

    fig.update_layout(
        title=dict(
            text="Faceted Constellation Pass Timeline - 12-Hour High-Density Analysis Window",
            font=dict(family="Serif", size=13, color="black"),
            x=0.5,
            y=0.97
        ),
        xaxis=dict(showgrid=True, gridcolor="rgba(210, 210, 210, 0.5)"),
        width=850,
        height=180 * len(stations) + 120,
        margin=dict(l=80, r=40, t=70, b=60),
        paper_bgcolor="white",
        plot_bgcolor="white"
    )

    for i in range(1, len(stations) + 1):
        x_axis_key = f"xaxis{i}" if i > 1 else "xaxis"
        y_axis_key = f"yaxis{i}" if i > 1 else "yaxis"

        if x_axis_key in fig.layout:
            fig.layout[x_axis_key].update(
                title=dict(text="Timeline Horizon (UTC)" if i == len(stations) else "", font=dict(family="Serif", size=11)),
                tickfont=dict(family="Serif", size=9, color="black"),
                showgrid=True,
                gridcolor="rgba(210, 210, 210, 0.5)"
            )
            
        if y_axis_key in fig.layout:
            fig.layout[y_axis_key].update(
                tickmode="array",
                tickvals=list(sat_y_map.values()),
                ticktext=list(sat_y_map.keys()),
                tickfont=dict(family="Serif", size=9, color="black"),
                showgrid=True,
                gridcolor="rgba(210, 210, 210, 0.3)",
                range=[-0.5, len(satellites) - 0.5]
            )

    fig.for_each_annotation(
        lambda a: a.update(
            font=dict(family="Serif", size=11, color="black", weight="bold")
        )
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(output_svg), format="svg")
    print(f"[SUCCESS] Faceted academic Gantt timeline safely compiled via Primitive GO and saved to: {output_svg.resolve()}")


if __name__ == "__main__":
    generate_academic_gantt()