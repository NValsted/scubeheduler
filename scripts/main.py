from dataclasses import dataclass

from astropy import units as u
from matplotlib import pyplot as plt
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from poliastro.util import Time

LAUNCH_DATE = Time("2022-07-11 05:05", scale="utc")
EARTH_MU = (3.986004418 * (10**14)) << (u.m**3 / u.s**2)  # type: ignore
MEAN_MOTION = 15.09 << 1 / u.day  # type: ignore

SEMI_MAJOR_AXIS = (EARTH_MU / (MEAN_MOTION**2)) ** (1 / 3)
ECCENTRICITY = 0.0 << u.one
INCLINATION = 97.566002 << u.deg  # type: ignore
RIGHT_ASCENSION = 0.0 << u.deg  # type: ignore
ARGUMENT_OF_PERIGEE = 0 << u.deg  # type: ignore
INITIAL_ANOMALY = -90.0 << u.deg  # type: ignore


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
    orbit = Orbit.from_classical(
        Earth,
        SEMI_MAJOR_AXIS,
        ECCENTRICITY,
        INCLINATION,
        RIGHT_ASCENSION,
        ARGUMENT_OF_PERIGEE,
        INITIAL_ANOMALY,
    )

    pos = []
    for i in range(int(60 * 10)):
        orbit = orbit.propagate(1 << u.minute)  # type: ignore
        pos.append(orbit.r)

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(  # CubeSat
        [p[0].to_value() for p in pos],
        [p[1].to_value() for p in pos],
        [p[2].to_value() for p in pos],
        alpha=[p / len(pos) for p in range(len(pos))],
    )
    ax.scatter(  # Earth
        [0],
        [0],
        [0],
        color="green",
        s=200,
    )
    plt.show()


if __name__ == "__main__":
    main()
