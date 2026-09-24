import math
from typing import List, Tuple
from src.core.datatypes import SatelliteConfig

__all__ = ["build_walker_constellation", "validate_leo_altitude"]

EARTH_RADIUS_KM = 6378.137
EARTH_MU = 398600.4418      
SECONDS_PER_DAY = 86400.0

MIN_LEO_ALTITUDE_KM = 300.0   
MAX_LEO_ALTITUDE_KM = 1000.0  


def validate_leo_altitude(altitude_km: float) -> float:
    """
    Validates that the altitude falls within an operational LEO regime (~90 to 105 min)
    and computes the corresponding Mean Motion in revolutions per day.

    Args:
        altitude_km: Geometric altitude above sea level in kilometers.

    Returns:
        Mean motion in revolutions per day.

    Raises:
        ValueError: If the altitude is outside the permitted strict LEO bounds.
    """
    if not (MIN_LEO_ALTITUDE_KM <= altitude_km <= MAX_LEO_ALTITUDE_KM):
        raise ValueError(
            f"Altitude {altitude_km} km is outside strict LEO limits. "
            f"Must be between {MIN_LEO_ALTITUDE_KM} km and {MAX_LEO_ALTITUDE_KM} km "
            f"(orbital periods of ~90 to 105 min)."
        )

    semi_major_axis = EARTH_RADIUS_KM + altitude_km
    period_seconds = 2.0 * math.pi * math.sqrt((semi_major_axis ** 3) / EARTH_MU)
    return SECONDS_PER_DAY / period_seconds


def _calculate_tle_checksum(line_without_checksum: str) -> int:
    """
    Calculates the modulo-10 checksum digit according to the official NORAD TLE specification.
    Numbers add their face value, hyphens '-' add 1, and all other characters add 0.
    """
    checksum = 0
    for char in line_without_checksum:
        if char.isdigit():
            checksum += int(char)
        elif char == "-":
            checksum += 1
    return checksum % 10


def _generate_synthetic_tle(
    sat_id: int,
    mean_motion: float,
    inc: float,
    raan: float,
    mean_anomaly: float
) -> Tuple[str, str]:
    """
    Constructs the two TLE lines strictly adhering to column indices
    and spacing constraints required by standard parsers and propagators.
    """
    sat_id_str = f"{sat_id % 100000:05d}"
    inc_str = f"{(inc % 180.0):8.4f}".rjust(8)
    raan_str = f"{(raan % 360.0):8.4f}".rjust(8)
    ma_str = f"{(mean_anomaly % 360.0):8.4f}".rjust(8)
    
    mm_str = f"{mean_motion:11.8f}"[:11].ljust(11)

    l1_base = f"1 {sat_id_str}U 26001A   26001.00000000  .00000000  00000-0  00000-0 0  999"
    c1 = _calculate_tle_checksum(l1_base)
    line1 = f"{l1_base}{c1}"

    l2_base = f"2 {sat_id_str} {inc_str} {raan_str} 0000000  00.0000 {ma_str} {mm_str}00001"
    c2 = _calculate_tle_checksum(l2_base)
    line2 = f"{l2_base}{c2}"

    return line1, line2


def build_walker_constellation(
    constellation_type: str,
    t: int,
    p: int,
    f: int,
    altitude_km: float = 550.0,
    inc: float = 53.0,
    base_id: int = 90000
) -> List[SatelliteConfig]:
    """
    Constructs a synthetic Walker constellation (Delta or Star).

    Args:
        constellation_type: "Walker Delta" or "Walker Star".
        t: Total number of satellites in the constellation.
        p: Number of orbital planes.
        f: Relative phasing parameter between adjacent planes (0 <= f < p).
        altitude_km: Circular orbital altitude above Earth in km (300 km to 1000 km).
        inc: Orbital inclination in degrees.
        base_id: Initial NORAD numerical tracking identifier for synthetic assets.

    Returns:
        A list of SatelliteConfig instances populated with valid LEO TLEs.

    Raises:
        ValueError: If constellation_type is unknown, if geometry constraints are violated,
                    or if the altitude falls outside valid LEO boundaries.
    """
    if constellation_type not in ("Walker Delta", "Walker Star"):
        raise ValueError(
            f"Unsupported constellation type: '{constellation_type}'. "
            f"Valid options: 'Walker Delta', 'Walker Star'."
        )

    if t <= 0 or p <= 0:
        raise ValueError("Total satellites 't' and orbital planes 'p' must be positive integers.")

    if t % p != 0:
        raise ValueError(
            f"Invalid Walker constellation: total satellites t={t} "
            f"must be evenly divisible by the number of planes p={p}."
        )

    if not (0 <= f < p):
        raise ValueError(
            f"Phasing factor 'f' must satisfy 0 <= f < p. Received: f={f}, p={p}."
        )

    mean_motion = validate_leo_altitude(altitude_km)

    sats_per_plane = t // p
    pu = 360.0 / t  
    
    raan_spread = 360.0 if constellation_type == "Walker Delta" else 180.0

    satellites = []
    current_sat_id = base_id

    for plane_idx in range(p):
        raan = (plane_idx * raan_spread / p) % 360.0
        phase_offset = plane_idx * f * pu

        for sat_idx in range(sats_per_plane):
            mean_anomaly = (sat_idx * (360.0 / sats_per_plane) + phase_offset) % 360.0

            l1, l2 = _generate_synthetic_tle(
                sat_id=current_sat_id,
                mean_motion=mean_motion,
                inc=inc,
                raan=raan,
                mean_anomaly=mean_anomaly
            )

            sat_label = f"SYNTH_{constellation_type.replace(' ', '_').upper()}_{current_sat_id}"

            satellites.append(
                SatelliteConfig(
                    norad_id=current_sat_id,
                    name=sat_label,
                    tle_line1=l1,
                    tle_line2=l2
                )
            )
            current_sat_id += 1

    return satellites