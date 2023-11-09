from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable

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


@dataclass
class ActionResponse:
    pass


class Scheduler:
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


class Satellite:
    spacecraft: adcssim.spacecraft.Spacecraft  # type: ignore
    orbit: Orbit
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
        spacecraft: adcssim.spacecraft.Spacecraft,  # type: ignore
        orbit: Orbit,
    ):
        self.spacecraft = spacecraft
        self.orbit = orbit
        self.time = 0 << u.s

        self.nominal_state_func = self.default_nominal_state_func
        self.perturbations_func = self.default_perturbations_func
        self.position_velocity_func = self.default_position_velocity_func

    @staticmethod
    def default_nominal_state_func(
        t: float,
        w_nominal: np.ndarray = np.array([1, 0, 0]),
        dcm_0_nominal: np.ndarray = np.eye(3),
    ):
        if w_nominal[0] != 0:
            dcm_nominal = np.matmul(
                adcssim.math_utils.t1_matrix(w_nominal[0] * t),  # type: ignore
                dcm_0_nominal,
            )
        elif w_nominal[1] != 0:
            dcm_nominal = np.matmul(
                adcssim.math_utils.t2_matrix(w_nominal[1] * t),  # type: ignore
                dcm_0_nominal,
            )
        elif w_nominal[2] != 0:
            dcm_nominal = np.matmul(
                adcssim.math_utils.t3_matrix(w_nominal[2] * t),  # type: ignore
                dcm_0_nominal,
            )
        else:
            dcm_nominal = dcm_0_nominal
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
        orbit_at_t = self.orbit.propagate(
            t - self.time.to(u.minute).to_value()  # type: ignore
        )
        return orbit_at_t.r.to(u.km).to_value(), orbit_at_t.v.to(u.km / u.s).to_value()

    def propagate(self, time: u.Quantity) -> None:
        assert time >= self.delta_t
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
        self.time += time << self.delta_t.unit  # type: ignore
        self.q_actual = result["q_actual"][0]

    @classmethod
    def factory(cls):
        sat_orbit = Orbit.from_classical(
            Earth,
            SEMI_MAJOR_AXIS,
            ECCENTRICITY,
            INCLINATION,
            RIGHT_ASCENSION,
            ARGUMENT_OF_PERIGEE,
            INITIAL_ANOMALY,
        )
        r_0 = np.array(sat_orbit.r.to(u.km))  # type: ignore
        v_0 = np.array(sat_orbit.v.to(u.km / u.s))  # type: ignore
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
            / sat_orbit.period.to(u.s).to_value()  # type: ignore
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
        dimensions = [10, 10, 10] << u.cm  # type: ignore
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
        return cls(spacecraft, sat_orbit)


def main():
    satellite: Satellite = Satellite.factory()

    eph = load("de421.bsp")
    earth, sun = eph["earth"], eph["sun"]

    ts = load.timescale()

    pos = []
    power = []
    q = []
    for i in tqdm(range(int(60 * 1))):
        crr_datetime = LAUNCH_DATE + timedelta(minutes=i)
        t = ts.utc(crr_datetime.year, crr_datetime.month, crr_datetime.day)
        satellite.propagate(1 << u.minute)  # type: ignore
        pos.append(satellite.orbit.r)
        q.append(satellite.q_actual)

        earth_pos = earth.at(t)  # type: ignore
        sun_pos = sun.at(t)  # type: ignore

        v_earth_to_sat = satellite.orbit.r
        v_earth_to_sun = sun_pos.position.to(u.km) - earth_pos.position.to(u.km)
        v_sat_to_earth = -v_earth_to_sat
        v_sat_to_sun = sun_pos.position.to(u.km) - satellite.orbit.r
        sat_direction = adcssim.math_utils.quaternion_to_dcm(  # type: ignore
            satellite.q_actual
        )[0]

        orbit_sun_angle = np.arccos(
            np.dot(
                v_earth_to_sat / np.linalg.norm(v_earth_to_sat),
                v_earth_to_sun / np.linalg.norm(v_earth_to_sun),
            )
        )
        shadow_angle_tresh = np.arctan(Earth.R / np.linalg.norm(v_earth_to_sat)) + (
            np.pi / 2 << u.rad
        )

        if False and orbit_sun_angle >= shadow_angle_tresh:
            sun_view_factor = 0
        else:
            sun_view_factor = np.dot(
                sat_direction / np.linalg.norm(sat_direction),
                v_sat_to_sun / np.linalg.norm(v_sat_to_sun),
            ).to_value()

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

    # plt.plot([q_i[0] for q_i in q if abs(q_i[0])])
    # plt.plot([q_i[1] for q_i in q if abs(q_i[1])])
    # plt.plot([q_i[2] for q_i in q if abs(q_i[2])])
    # plt.plot([q_i[3] for q_i in q if abs(q_i[3])])

    plt.show()


if __name__ == "__main__":
    main()
