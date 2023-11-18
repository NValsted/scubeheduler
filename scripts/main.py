from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import partial
from typing import Any, Callable

import numpy as np
from astropy import units as u
from matplotlib import pyplot as plt
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from skyfield.api import load
from tqdm import tqdm

from lib import adcssim

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

ENABLE_ADCS_SIM = False


def dot_between(v1: np.ndarray, v2: np.ndarray) -> float:
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


class WorldQuery:
    ts: Any  # TODO : type
    earth: Any  # TODO : type
    sun: Any  # TODO : type
    sat_orbit: Orbit
    launch_time: datetime
    current_time: datetime

    def __init__(
        self,
        ts: Any = load.timescale(),
        earth: Any = None,
        sun: Any = None,
        sat_orbit: Orbit = Orbit.from_classical(
            Earth,
            SEMI_MAJOR_AXIS,
            ECCENTRICITY,
            INCLINATION,
            RIGHT_ASCENSION,
            ARGUMENT_OF_PERIGEE,
            INITIAL_ANOMALY,
        ),
        launch_time: datetime = LAUNCH_DATE,
    ):
        if earth is None or sun is None:
            eph = load("de421.bsp")
            earth, sun = eph["earth"], eph["sun"]
        self.ts = ts
        self.earth = earth
        self.sun = sun
        self.sat_orbit = sat_orbit
        self.launch_time = launch_time
        self.current_time = launch_time

    def update_time(self, new_time: datetime):
        self.sat_orbit = self.sat_orbit.propagate(
            (new_time - self.current_time).total_seconds() << u.s
        )
        self.current_time = new_time


@dataclass
class SolarPanel:
    direction: np.ndarray
    surface_area: u.Quantity


class Satellite:
    world_query: WorldQuery
    spacecraft: adcssim.spacecraft.Spacecraft  # type: ignore
    solar_panels: list[SolarPanel]
    time: u.Quantity
    delta_t: u.Quantity = 1 << u.minute  # type: ignore
    q_actual: np.ndarray

    nominal_state_func: Callable[
        [float],
        tuple[np.ndarray, np.ndarray],
    ]
    perturbations_func: Callable[
        [adcssim.spacecraft.Spacecraft],  # type: ignore
        np.ndarray,
    ]
    position_velocity_func: Callable[
        [float],
        tuple[np.ndarray, np.ndarray],
    ]

    def __init__(
        self,
        world_query: WorldQuery,
        spacecraft: adcssim.spacecraft.Spacecraft,  # type: ignore
        solar_panels: list[SolarPanel],
    ):
        self.spacecraft = spacecraft
        self.world_query = world_query
        self.solar_panels = solar_panels
        self.time = 0 << u.minute  # type: ignore

        self.nominal_state_func = partial(
            self.default_nominal_state_func, satellite=self
        )  # type: ignore
        self.perturbations_func = self.default_perturbations_func
        self.position_velocity_func = self.default_position_velocity_func

    @staticmethod
    def default_nominal_state_func(t: float, satellite: "Satellite"):
        w_nominal = satellite.spacecraft.w
        sky_field_t = satellite.world_query.ts.utc(
            satellite.world_query.current_time.year,
            satellite.world_query.current_time.month,
            satellite.world_query.current_time.day,
        )
        dcm_nominal = [
            satellite.world_query.sun.at(sky_field_t).position.to(u.km).to_value()
            - satellite.world_query.sat_orbit.r.to(u.km).to_value(),
            satellite.world_query.earth.at(sky_field_t).position.to(u.km).to_value()
            - satellite.world_query.sat_orbit.r.to(u.km).to_value(),
            satellite.world_query.sat_orbit.r.to(u.km).to_value(),
        ]
        return dcm_nominal, w_nominal

    @staticmethod
    def default_perturbations_func(
        spacecraft: adcssim.spacecraft.Spacecraft,  # type: ignore
    ):
        return (
            spacecraft.approximate_gravity_gradient_torque()
            + spacecraft.approximate_magnetic_field_torque()
        )

    def default_position_velocity_func(self, t: float):
        orbit_at_t = self.world_query.sat_orbit.propagate(
            t - self.time.to(u.minute).to_value()  # type: ignore
        )
        return (
            orbit_at_t.r.to(u.km).to_value(),
            orbit_at_t.v.to(u.km / u.minute).to_value(),  # type: ignore
        )

    def propagate(self, time: u.Quantity) -> None:
        assert time >= self.delta_t
        if ENABLE_ADCS_SIM:
            result = adcssim.simulation.simulate_adcs(  # type: ignore
                satellite=self.spacecraft,
                nominal_state_func=self.nominal_state_func,
                perturbations_func=self.perturbations_func,
                position_velocity_func=self.position_velocity_func,
                start_time=self.time.to(self.delta_t.unit).to_value(),
                delta_t=self.delta_t.to_value(),
                stop_time=(self.time + time)
                .to(self.delta_t.unit)  # type: ignore
                .to_value()
                - 1,
                verbose=False,
            )
            self.q_actual = result["q_actual"][0]
        else:
            dcm_nominal, _ = self.nominal_state_func(
                self.time.to(self.delta_t.unit).to_value()  # type: ignore
            )
            self.q_actual = adcssim.math_utils.dcm_to_quaternion(  # type: ignore
                np.array(dcm_nominal)
            )
        self.time += time << self.delta_t.unit  # type: ignore

    @classmethod
    def factory(cls, world_query: WorldQuery):
        r_0 = np.array(world_query.sat_orbit.r.to(u.km))  # type: ignore
        v_0 = np.array(world_query.sat_orbit.v.to(u.km / u.minute))  # type: ignore
        b_x = -adcssim.math_utils.normalize(r_0)  # type: ignore
        b_y = adcssim.math_utils.normalize(v_0)  # type: ignore
        b_z = adcssim.math_utils.cross(b_x, b_y)  # type: ignore

        dcm_0_nominal = np.stack([b_x, b_y, b_z])
        q_0_nominal = adcssim.math_utils.dcm_to_quaternion(  # type: ignore
            dcm_0_nominal
        )
        w_nominal_i = (
            2
            * np.pi
            / world_query.sat_orbit.period.to(u.minute).to_value()  # type: ignore
            * adcssim.math_utils.normalize(  # type: ignore
                adcssim.math_utils.cross(r_0, v_0)  # type: ignore
            )
        )
        w_nominal = np.matmul(dcm_0_nominal, w_nominal_i)

        # provide some initial offset in both the attitude and angular velocity
        q_0 = adcssim.math_utils.quaternion_multiply(  # type: ignore
            np.array([0, np.sin(2 * np.pi / 180 / 2), 0, np.cos(2 * np.pi / 180 / 2)]),
            q_0_nominal,
        )
        w_0 = w_nominal + np.array([0.005, 0, 0])

        mass = 5.7 << u.kg  # type: ignore
        dimensions = [10, 10, 30] << u.cm  # type: ignore
        moment_of_inertia = (
            1
            / 12
            * mass.to(u.kg).to_value()  # type: ignore
            * np.diag(
                [
                    dimensions[1].to(u.m).to_value() ** 2
                    + dimensions[2].to(u.m).to_value() ** 2,
                    dimensions[0].to(u.m).to_value() ** 2
                    + dimensions[2].to(u.m).to_value() ** 2,
                    dimensions[0].to(u.m).to_value() ** 2
                    + dimensions[1].to(u.m).to_value() ** 2,
                ]
            )
        )

        controller = adcssim.controller.PDController(  # type: ignore
            k_d=np.diag([0.01, 0.01, 0.01]), k_p=np.diag([0.1, 0.1, 0.1])
        )

        gyros = adcssim.sensors.Gyros(  # type: ignore
            bias_stability=1, angular_random_walk=0.07
        )
        magnetometer = adcssim.sensors.Magnetometer(resolution=10e-9)  # type: ignore
        earth_horizon_sensor = adcssim.sensors.EarthHorizonSensor(  # type: ignore
            accuracy=0.25
        )

        actuators = adcssim.actuators.Actuators(  # type: ignore
            rxwl_mass=226e-3,
            rxwl_radius=0.5 * 65e-3,
            rxwl_max_torque=20e-3,
            rxwl_max_momentum=0.18,
            noise_factor=0.03,
        )

        spacecraft = adcssim.spacecraft.Spacecraft(  # type: ignore
            J=moment_of_inertia,
            controller=controller,
            gyros=gyros,
            magnetometer=magnetometer,
            earth_horizon_sensor=earth_horizon_sensor,
            actuators=actuators,
            q=q_0,
            w=w_0,
            r=r_0,
            v=v_0,
        )

        solar_panels = [
            SolarPanel(
                direction=np.array([1, 0, 0]),
                surface_area=0.1 << u.m**2,  # type: ignore
            )
        ]
        return cls(
            world_query=world_query, spacecraft=spacecraft, solar_panels=solar_panels
        )


@dataclass
class SimStatsPoint:
    time: datetime
    power: u.Quantity


class SimHandler:
    satellite: Satellite
    world_query: WorldQuery
    launch_time: datetime
    sim_stats: list[SimStatsPoint]

    def __init__(
        self,
        satellite: Satellite,
        world_query: WorldQuery,
    ):
        self.satellite = satellite
        self.world_query = world_query
        self.launch_time = world_query.launch_time
        self.sim_stats = []

    def calculate_power(self) -> u.Quantity:
        sky_field_t = self.world_query.ts.utc(
            self.world_query.current_time.year,
            self.world_query.current_time.month,
            self.world_query.current_time.day,
        )

        earth_pos = self.world_query.earth.at(sky_field_t)
        sun_pos = self.world_query.sun.at(sky_field_t)

        v_earth_to_sat = self.world_query.sat_orbit.r
        v_earth_to_sun = sun_pos.position.to(u.km) - earth_pos.position.to(u.km)
        v_sat_to_sun = sun_pos.position.to(u.km) - self.world_query.sat_orbit.r
        sat_direction = adcssim.math_utils.quaternion_to_dcm(  # type: ignore
            self.satellite.q_actual
        )

        cum_power = 0 << u.W  # type: ignore
        for panel in self.satellite.solar_panels:  # TODO: use panel detais
            sun_view_factor = dot_between(sat_direction[0], v_sat_to_sun)
            if sun_view_factor < 0:
                sun_view_factor = abs(sun_view_factor / 2)

            shadow_thresh = ((np.pi / 2) << u.rad) - np.arccos(
                1
                - (np.linalg.norm(v_earth_to_sat) - Earth.R)
                / np.linalg.norm(v_earth_to_sat)
            )
            if dot_between(v_earth_to_sat, -v_earth_to_sun) > np.cos(shadow_thresh):
                sun_view_factor = 0

            cum_power += (
                SOLAR_IRRADIANCE
                * SOLAR_CELL_EFFICIENCY
                * SOLAR_PANEL_AREA
                * sun_view_factor
            )

        return cum_power

    def propagate(self, dt: u.Quantity) -> None:
        self.world_query.update_time(
            self.world_query.current_time
            + timedelta(minutes=dt.to(u.minute).to_value())  # type: ignore
        )
        self.satellite.propagate(dt)
        assert (
            self.satellite.time.to(u.s).to_value()
            == (self.world_query.current_time - self.launch_time).total_seconds()
        )

        power = self.calculate_power()
        self.sim_stats.append(
            SimStatsPoint(time=self.world_query.current_time, power=power)
        )


def main():
    world_query = WorldQuery()
    satellite: Satellite = Satellite.factory(world_query=world_query)
    sim_handler: SimHandler = SimHandler(satellite=satellite, world_query=world_query)

    for _ in tqdm(range(int(60 * 24 * 2))):
        sim_handler.propagate(1 << u.minute)  # type: ignore

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

    plt.plot([p.power.to_value() for p in sim_handler.sim_stats])

    # plt.plot([q_i[0] for q_i in q if abs(q_i[0])])
    # plt.plot([q_i[1] for q_i in q if abs(q_i[1])])
    # plt.plot([q_i[2] for q_i in q if abs(q_i[2])])
    # plt.plot([q_i[3] for q_i in q if abs(q_i[3])])

    plt.show()


if __name__ == "__main__":
    main()
