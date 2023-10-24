from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np
from astropy import units as u
from matplotlib import pyplot as plt
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from skyfield.api import load
from tqdm import tqdm

LAUNCH_DATE = datetime(2024, 7, 1, tzinfo=timezone.utc)

EARTH_MU = (3.986004418 * (10**14)) << (u.m**3 / u.s**2)  # type: ignore
MEAN_MOTION = 15.09 << 1 / u.day  # type: ignore

SEMI_MAJOR_AXIS = (EARTH_MU / (MEAN_MOTION**2)) ** (1 / 3)
ECCENTRICITY = 0.0 << u.one
INCLINATION = 97.566002 << u.deg  # type: ignore
RIGHT_ASCENSION = 0.0 << u.deg  # type: ignore
ARGUMENT_OF_PERIGEE = 0 << u.deg  # type: ignore
INITIAL_ANOMALY = -90.0 << u.deg  # type: ignore

SOLAR_IRRADIANCE = 1361 << u.W / u.m**2  # type: ignore
SOLAR_CELL_EFFICIENCY = 0.293 << u.one  # type: ignore
SOLAR_PANEL_AREA = (10**-4) * 10 * 3 * 4 << u.m**2  # type: ignore


@dataclass
class ActionResponse:
    pass


class Scheduler:
    # sun_vector = None

    def __init__(self):
        pass

    def update(self, response: ActionResponse):
        # measured power while executing action to dynamically update model?
        # query fine sun sensor to get sun vector ?
        # query attitude ?
        pass

    def get_plan(self):
        pass


class Battery:
    pass


class SolarPanels:
    pass


class Satellite:
    pass


def main():
    sat_orbit = Orbit.from_classical(
        Earth,
        SEMI_MAJOR_AXIS,
        ECCENTRICITY,
        INCLINATION,
        RIGHT_ASCENSION,
        ARGUMENT_OF_PERIGEE,
        INITIAL_ANOMALY,
    )

    eph = load("de421.bsp")
    earth, sun = eph["earth"], eph["sun"]

    ts = load.timescale()

    pos = []
    power = []
    for i in tqdm(range(int(60 * 10))):
        crr_datetime = LAUNCH_DATE + timedelta(minutes=i)
        t = ts.utc(crr_datetime.year, crr_datetime.month, crr_datetime.day)
        sat_orbit = sat_orbit.propagate(1 << u.minute)  # type: ignore
        pos.append(sat_orbit.r)

        earth_pos = earth.at(t)  # type: ignore
        sun_pos = sun.at(t)  # type: ignore

        v_earth_to_sat = sat_orbit.r
        v_earth_to_sun = sun_pos.position.to(u.km) - earth_pos.position.to(u.km)
        v_sat_to_earth = -v_earth_to_sat
        v_sat_to_sun = sun_pos.position.to(u.km) - sat_orbit.r
        sat_direction = v_sat_to_earth  # satellite always pointing towards earth

        orbit_sun_angle = np.arccos(
            np.dot(
                v_earth_to_sat / np.linalg.norm(v_earth_to_sat),
                v_earth_to_sun / np.linalg.norm(v_earth_to_sun),
            )
        )
        shadow_angle_tresh = np.arctan(Earth.R / np.linalg.norm(v_earth_to_sat)) + (
            np.pi / 2 << u.rad
        )

        if orbit_sun_angle >= shadow_angle_tresh:
            sun_view_factor = 0
        else:
            sun_view_factor = max(
                np.dot(
                    sat_direction / np.linalg.norm(sat_direction),
                    v_sat_to_sun / np.linalg.norm(v_sat_to_sun),
                ).to_value(),
                0,
            )

        power.append(
            SOLAR_IRRADIANCE
            * SOLAR_CELL_EFFICIENCY
            * SOLAR_PANEL_AREA
            * sun_view_factor
        )

    # fig = plt.figure()
    # ax = fig.add_subplot(projection="3d")
    # ax.scatter(  # CubeSat
    #     [p[0].to_value() for p in pos],
    #     [p[1].to_value() for p in pos],
    #     [p[2].to_value() for p in pos],
    #     alpha=[p / len(pos) for p in range(len(pos))],
    # )
    # ax.scatter(  # Earth
    #     [0],
    #     [0],
    #     [0],
    #     color="green",
    #     s=200,
    # )
    plt.plot([p.to_value() for p in power])
    plt.show()


if __name__ == "__main__":
    main()
