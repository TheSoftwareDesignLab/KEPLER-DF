import json
import pathlib
from typing import Optional, List
import numpy as np

__all__ = ["generate_kepler_kml"]


def _resolve_path(path_str: str) -> pathlib.Path:
    p = pathlib.Path(path_str)
    if p.is_absolute():
        return p
    if p.exists():
        return p.resolve()
    return (pathlib.Path(__file__).resolve().parent.parent / path_str).resolve()


def _interpolate_orbit_points(s_lat: float, s_lon: float, e_lat: float, e_lon: float, num_samples: int = 35) -> str:
    lats = np.linspace(s_lat, e_lat, num_samples)
    if abs(e_lon - s_lon) > 180.0:
        if s_lon > 0:
            lons = np.linspace(s_lon, e_lon + 360.0, num_samples)
            lons = np.where(lons > 180.0, lons - 360.0, lons)
        else:
            lons = np.linspace(s_lon + 360.0, e_lon, num_samples)
            lons = np.where(lons > 180.0, lons - 360.0, lons)
    else:
        lons = np.linspace(s_lon, e_lon, num_samples)
        
    coord_strings = []
    for lat, lon in zip(lats, lons):
        coord_strings.append(f"{lon},{lat},0")
    return " ".join(coord_strings)


def generate_kepler_kml(
    scenario_report_path: str = "data/constellation_dataset_phi4_14b_v2/scenario_5/scenario_report.json",
    physics_report_path: str = "data/constellation_dataset_phi4_14b_v2/scenario_5/physics_passes_report.json",
    output_kml_path: str = "utilities/output/kepler_simulation5.kml",
    selected_satellite_id: Optional[int] = None
) -> None:
    scenario_path = _resolve_path(scenario_report_path)
    physics_path = _resolve_path(physics_report_path)
    output_path = _resolve_path(output_kml_path)

    if not scenario_path.exists() or not physics_path.exists():
        print(f"[ERROR] Missing report files for KML generation.")
        return

    with open(scenario_path, "r", encoding="utf-8") as f:
        scenario_data = json.load(f)
    with open(physics_path, "r", encoding="utf-8") as f:
        physics_data = json.load(f)

    ground_stations = scenario_data.get("ground_stations", [])
    targets = scenario_data.get("targets", [])
    infra_passes = physics_data.get("infrastructure_passes", [])
    target_passes = physics_data.get("target_passes", [])

    all_sat_ids = sorted(list(set(
        [p.get("satellite_id") for p in infra_passes if p.get("satellite_id")] +
        [p.get("satellite_id") for p in target_passes if p.get("satellite_id")]
    )))

    kml_lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<kml xmlns="http://www.opengis.net/kml/2.2">',
        '<Document>',
        '  <name>Kepler Constellation Mission Control - All Satellites</name>',
        '  <Style id="station_style">',
        '    <IconStyle>',
        '      <color>ff0000ff</color>',
        '      <scale>1.4</scale>',
        '      <Icon><href>http://maps.google.com/mapfiles/kml/shapes/track.png</href></Icon>',
        '    </IconStyle>',
        '    <LabelStyle>',
        '      <color>ff0000ff</color>',
        '      <scale>0.9</scale>',
        '    </LabelStyle>',
        '  </Style>',
        '  <Style id="target_point_style">',
        '    <IconStyle>',
        '      <color>ff00ff00</color>',
        '      <scale>1.2</scale>',
        '      <Icon><href>http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png</href></Icon>',
        '    </IconStyle>',
        '    <LabelStyle>',
        '      <color>ff00ff00</color>',
        '      <scale>0.9</scale>',
        '    </LabelStyle>',
        '  </Style>',
        '  <Style id="generation_bounding_box_style">',
        '    <LineStyle>',
        '      <color>ff00ff00</color>',
        '      <width>3.0</width>',
        '    </LineStyle>',
        '    <PolyStyle>',
        '      <color>4000ff00</color>',
        '      <fill>1</fill>',
        '      <outline>1</outline>',
        '    </PolyStyle>',
        '    <LabelStyle>',
        '      <color>ff00ff00</color>',
        '      <scale>0.95</scale>',
        '    </LabelStyle>',
        '  </Style>',
        '  <Style id="infra_line_style">',
        '    <LineStyle>',
        '      <color>ffc000ff</color>',
        '      <width>4.0</width>',
        '    </LineStyle>',
        '  </Style>',
        '  <Style id="target_line_style">',
        '    <LineStyle>',
        '      <color>ffffa500</color>',
        '      <width>4.0</width>',
        '    </LineStyle>',
        '    <LabelStyle>',
        '      <color>ffffa500</color>',
        '      <scale>0.9</scale>',
        '    </LabelStyle>',
        '  </Style>',
        '  <Folder>',
        '    <name>Ground Stations</name>'
    ]

    for gs in ground_stations:
        gs_name = gs.get("name", gs.get("id", "Unknown"))
        lat = gs.get("latitude", 0.0)
        lon = gs.get("longitude", 0.0)
        alt = gs.get("elevation", 0.0)
        
        kml_lines.extend([
            '    <Placemark>',
            f'      <name>GS: {gs_name}</name>',
            '      <styleUrl>#station_style</styleUrl>',
            '      <Point>',
            '        <altitudeMode>clampToGround</altitudeMode>',
            f'        <coordinates>{lon},{lat},{alt}</coordinates>',
            '      </Point>',
            '    </Placemark>'
        ])

    kml_lines.extend([
        '  </Folder>',
        '  <Folder>',
        '    <name>Generation Bounding Boxes (Dataset Areas)</name>'
    ])

    for tgt in targets:
        task_id = tgt.get("task_id", "Unknown")
        region_tag = tgt.get("region_tag", task_id)
        task_type = tgt.get("task_type", "polygon")
        coords = tgt.get("coordinates", [])

        if not coords:
            continue

        if task_type == "polygon" or len(coords) > 1:
            coord_strings = []
            raw_lats = []
            raw_lons = []
            for pt in coords:
                coord_strings.append(f"{pt[1]},{pt[0]},0")
                raw_lats.append(pt[0])
                raw_lons.append(pt[1])
            if coords[0] != coords[-1]:
                coord_strings.append(f"{coords[0][1]},{coords[0][0]},0")
            coord_block = " ".join(coord_strings)
            
            c_lat = sum(raw_lats) / len(raw_lats)
            c_lon = sum(raw_lons) / len(raw_lons)

            # CORRECCIÓN: Separamos el marcador de texto del polígono físico en dos Placemarks independientes
            # para forzar a Google Earth a renderizar el ícono central en verde brillante y evitar el bug del pin azul.
            kml_lines.extend([
                '    <Placemark>',
                f'      <name>{task_id}</name>',
                f'      <description>Region Tag: {region_tag} (Polygon Centroid)</description>',
                '      <styleUrl>#target_point_style</styleUrl>',
                '      <Point>',
                '        <altitudeMode>clampToGround</altitudeMode>',
                f'        <coordinates>{c_lon},{c_lat},0</coordinates>',
                '      </Point>',
                '    </Placemark>',
                '    <Placemark>',
                f'      <name>{task_id} Boundary</name>',
                f'      <description>Region Tag: {region_tag} (Polygon Area Bound)</description>',
                '      <styleUrl>#generation_bounding_box_style</styleUrl>',
                '      <Polygon>',
                '        <tessellate>1</tessellate>',
                '        <altitudeMode>clampToGround</altitudeMode>',
                '        <outerBoundaryIs>',
                '            <LinearRing>',
                f'              <coordinates>{coord_block}</coordinates>',
                '            </LinearRing>',
                '        </outerBoundaryIs>',
                '      </Polygon>',
                '    </Placemark>'
            ])
        else:
            pt_lat = coords[0][0]
            pt_lon = coords[0][1]
            kml_lines.extend([
                '    <Placemark>',
                f'      <name>{task_id}</name>',
                f'      <description>Region Tag: {region_tag} (Spot Location)</description>',
                '      <styleUrl>#target_point_style</styleUrl>',
                '      <Point>',
                '        <altitudeMode>clampToGround</altitudeMode>',
                f'        <coordinates>{pt_lon},{pt_lat},0</coordinates>',
                '      </Point>',
                '    </Placemark>'
            ])

    kml_lines.append('  </Folder>')

    for sat_id in all_sat_ids:
        kml_lines.extend([
            f'  <Folder>',
            f'    <name>Satellite {sat_id}</name>',
            f'    <Folder>',
            f'      <name>Sat {sat_id} - Downlink Tracks</name>'
        ])

        for p in infra_passes:
            if p.get("satellite_id") != sat_id:
                continue
                
            aos_str = p.get("aos_utc")
            if not aos_str:
                continue

            gs_id = p.get("ground_station_id")
            los = p.get("los_utc")
            duration = p.get("duration_s", 0)
            max_el = p.get("max_el_deg", 0.0)
            r_aos = p.get("range_aos_km", 0.0)
            r_los = p.get("range_los_km", 0.0)
            cap = p.get("estimated_transmission_capacity_mb", 0.0)
            
            s_lat = p.get("subsat_start_lat")
            s_lon = p.get("subsat_start_lon")
            e_lat = p.get("subsat_end_lat")
            e_lon = p.get("subsat_end_lon")

            if None in (s_lat, s_lon, e_lat, e_lon):
                continue

            coord_block = _interpolate_orbit_points(s_lat, s_lon, e_lat, e_lon, num_samples=35)

            kml_lines.extend([
                '      <Placemark>',
                f'        <name>Downlink → {gs_id}</name>',
                '        <ExtendedData>',
                f'          <Data name="Satellite NORAD"><value>{sat_id}</value></Data>',
                f'          <Data name="Ground Station"><value>{gs_id}</value></Data>',
                f'          <Data name="AOS UTC"><value>{aos_str}</value></Data>',
                f'          <Data name="LOS UTC"><value>{los}</value></Data>',
                f'          <Data name="Duration"><value>{duration} s</value></Data>',
                f'          <Data name="Max Elevation"><value>{max_el:.2f}°</value></Data>',
                f'          <Data name="Slant Range AOS"><value>{r_aos:.1f} km</value></Data>',
                f'          <Data name="Slant Range LOS"><value>{r_los:.1f} km</value></Data>',
                f'          <Data name="Downlink Capacity"><value>{cap:.1f} Mb</value></Data>',
                '        </ExtendedData>',
                '        <styleUrl>#infra_line_style</styleUrl>',
                '        <LineString>',
                '          <tessellate>1</tessellate>',
                '          <altitudeMode>clampToGround</altitudeMode>',
                '          <coordinates>',
                f'            {coord_block}',
                '          </coordinates>',
                '        </LineString>',
                '      </Placemark>'
            ])

        kml_lines.extend([
            f'    </Folder>',
            f'    <Folder>',
            f'      <name>Sat {sat_id} - Observation Tracks</name>'
        ])

        for p in target_passes:
            if p.get("satellite_id") != sat_id:
                continue
                
            aos_str = p.get("aos_utc")
            if not aos_str:
                continue

            task_id = p.get("task_id")
            region = p.get("region_tag")
            los = p.get("los_utc")
            v_dur = p.get("visibility_duration_s", 0)
            i_dur = p.get("sensor_imaging_duration_s", 0)
            max_el = p.get("max_el_deg", 0.0)
            data_vol = p.get("estimated_onboard_data_generation_mb", 0.0)
            
            s_lat = p.get("subsat_start_lat")
            s_lon = p.get("subsat_start_lon")
            e_lat = p.get("subsat_end_lat")
            e_lon = p.get("subsat_end_lon")

            if None in (s_lat, s_lon, e_lat, e_lon):
                continue

            coord_block = _interpolate_orbit_points(s_lat, s_lon, e_lat, e_lon, num_samples=35)
            mid_idx = 17
            coords_list = coord_block.split(" ")
            label_coord = coords_list[mid_idx] if mid_idx < len(coords_list) else coords_list[0]

            kml_lines.extend([
                '      <Placemark>',
                f'        <name>Pass: Imaging Target [{task_id}]</name>',
                '        <ExtendedData>',
                f'          <Data name="Satellite NORAD"><value>{sat_id}</value></Data>',
                f'          <Data name="Task ID"><value>{task_id}</value></Data>',
                f'          <Data name="Region Tag"><value>{region}</value></Data>',
                f'          <Data name="AOS UTC"><value>{aos_str}</value></Data>',
                f'          <Data name="LOS UTC"><value>{los}</value></Data>',
                f'          <Data name="Visibility Window"><value>{v_dur} s</value></Data>',
                f'          <Data name="Active Imaging"><value>{i_dur} s</value></Data>',
                f'          <Data name="Max Elevation"><value>{max_el:.2f}°</value></Data>',
                f'          <Data name="Payload Generation"><value>{data_vol:.1f} Mb</value></Data>',
                '        </ExtendedData>',
                '        <styleUrl>#target_line_style</styleUrl>',
                '        <MultiGeometry>',
                '          <Point>',
                '            <altitudeMode>clampToGround</altitudeMode>',
                f'            <coordinates>{label_coord}</coordinates>',
                '          </Point>',
                '          <LineString>',
                '            <tessellate>1</tessellate>',
                '            <altitudeMode>clampToGround</altitudeMode>',
                '            <coordinates>',
                f'              {coord_block}',
                '            </coordinates>',
                '          </LineString>',
                '        </MultiGeometry>',
                '      </Placemark>'
            ])

        kml_lines.extend([
            '    </Folder>',
            '  </Folder>'
        ])

    kml_lines.extend([
        '</Document>',
        '</kml>'
    ])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(kml_lines))

    print(f"[SUCCESS] High-contrast KML file with separated satellite hierarchies compiled at: {output_path.resolve()}")


if __name__ == "__main__":
    generate_kepler_kml()