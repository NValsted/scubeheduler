from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any

import numpy as np
from astropy import units as u
from matplotlib import pyplot as plt
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from scipy.spatial.transform import Rotation
from skyfield.api import load
from tqdm import tqdm

LAUNCH_DATE = datetime(2024, 7, 1, tzinfo=timezone.utc)

EARTH_MU = (3.986004418 * (10**14)) << (u.m**3 / u.s**2)  # type: ignore
MEAN_MOTION = (15.09 * 2 * np.pi) << (1 / u.day)  # type: ignore

SEMI_MAJOR_AXIS = (EARTH_MU / (MEAN_MOTION**2)) ** (1 / 3)
ECCENTRICITY = 0.0 << u.one
INCLINATION = 97.566002 << u.deg  # type: ignore
RIGHT_ASCENSION = 0.0 << u.deg  # type: ignore
ARGUMENT_OF_PERIGEE = 0 << u.deg  # type: ignore
INITIAL_ANOMALY = -90.0 << u.deg  # type: ignore

SOLAR_IRRADIANCE = 1361 << u.W / u.m**2  # type: ignore
SOLAR_CELL_EFFICIENCY = 0.293 << u.one  # type: ignore

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


class Battery:
    charge_level: u.Quantity
    capacity: u.Quantity
    efficiency: u.Quantity

    class DischargeError(Exception):
        pass

    def __init__(
        self,
        capacity: u.Quantity,
        efficiency: u.Quantity = 0.85 << u.one,  # type: ignore
        charge_level: u.Quantity = 0 << u.W * u.h,  # type: ignore
    ):
        self.capacity = capacity
        self.efficiency = efficiency
        self.charge_level = charge_level

    def charge(self, power: u.Quantity, time: u.Quantity):
        self.charge_level = min(
            power * self.efficiency * time + self.charge_level,  # type: ignore
            self.capacity,  # type: ignore
        )

    def discharge(self, power: u.Quantity, time: u.Quantity):
        discharge_energy = power / self.efficiency * time
        if discharge_energy > self.charge_level:
            raise Battery.DischargeError(
                f"Not enough charge: {self.charge_level=}-{discharge_energy=}"
            )

        self.charge_level = max(
            self.charge_level - discharge_energy,  # type: ignore
            0 << u.Wh,  # type: ignore
        )


class AttitudeObjective(Enum):
    EARTH = "EARTH"
    SUN = "SUN"


class Satellite:
    world_query: WorldQuery
    solar_panels: list[SolarPanel]
    battery: Battery
    attitude: Rotation
    objective: AttitudeObjective = AttitudeObjective.EARTH

    def __init__(
        self,
        world_query: WorldQuery,
        solar_panels: list[SolarPanel],
        battery: Battery,
    ):
        self.world_query = world_query
        self.solar_panels = solar_panels
        self.battery = battery

    def propagate(self, time: u.Quantity) -> None:
        if ENABLE_ADCS_SIM:
            raise NotImplementedError
        else:
            sky_field_t = self.world_query.ts.utc(
                self.world_query.current_time.year,
                self.world_query.current_time.month,
                self.world_query.current_time.day,
                self.world_query.current_time.hour,
                self.world_query.current_time.minute,
            )

            if self.objective == AttitudeObjective.SUN:
                sat_to_sun = self.world_query.sun.at(sky_field_t).position.to(u.km) - (
                    self.world_query.sat_orbit.r
                    + self.world_query.earth.at(sky_field_t).position.to(u.km)
                )
                sun_dir = sat_to_sun / np.linalg.norm(sat_to_sun)
                x_dir = sun_dir

            elif self.objective == AttitudeObjective.EARTH:
                sat_to_earth = -self.world_query.sat_orbit.r
                nadir = sat_to_earth / np.linalg.norm(sat_to_earth)
                x_dir = -nadir

            else:
                raise NotImplementedError

            objective_orbit_perp = np.cross(self.world_query.sat_orbit.v, x_dir)
            y_dir = objective_orbit_perp / np.linalg.norm(objective_orbit_perp)
            z_dir = np.cross(x_dir, y_dir)

            # Assert orthogonality
            assert abs(np.dot(x_dir, y_dir)) < 1e-15
            assert abs(np.dot(x_dir, z_dir)) < 1e-15
            assert abs(np.dot(y_dir, z_dir)) < 1e-15

            nominal_dcm = np.column_stack((x_dir, y_dir, z_dir))
            self.attitude = Rotation.from_matrix(nominal_dcm)

            if self.objective == AttitudeObjective.SUN:
                self.attitude = (
                    Rotation.from_rotvec(np.radians(45) * z_dir) * self.attitude
                )

    @classmethod
    def factory(cls, world_query: WorldQuery):
        # Factory for DISCO-II reference mission
        solar_panels = [
            SolarPanel(
                direction=np.array([1, 0, 0]),
                surface_area=8 * 30.18 << u.cm**2,  # type: ignore
            ),
            SolarPanel(
                direction=np.array([0, -1, 0]),
                surface_area=8 * 30.18 << u.cm**2,  # type: ignore
            ),
            SolarPanel(
                direction=np.array([-1 / np.sqrt(2), 1 / np.sqrt(2), 0]),
                surface_area=16 * 30.18 << u.cm**2,  # type: ignore
            ),
            SolarPanel(
                direction=np.array([1 / np.sqrt(2), -1 / np.sqrt(2), 0]),
                surface_area=16 * 30.18 << u.cm**2,  # type: ignore
            ),
        ]
        battery: Battery = Battery(capacity=92 << u.W * u.h)  # type: ignore
        return cls(world_query=world_query, solar_panels=solar_panels, battery=battery)


@dataclass
class SimStatsPoint:
    time: datetime
    power: u.Quantity
    orbit_alignment: float


@dataclass
class Task:
    _TASKS = []

    task_id: int
    priority: int
    duration: u.Quantity
    power: u.Quantity
    dependencies: list[int]

    @classmethod
    def random(cls):
        task_id = np.random.randint(0, 1000)
        dependencies = (
            np.random.choice(
                Task._TASKS,
                size=np.random.randint(0, min(10, len(Task._TASKS))),
                replace=False,
            )
            if len(Task._TASKS) > 0
            else []
        )
        Task._TASKS.append(task_id)
        return cls(
            task_id=task_id,
            priority=np.random.randint(0, 100),
            duration=np.random.randint(0, 50) << u.minute,  # type: ignore
            power=np.random.randint(0, 100) << u.W,  # type: ignore
            dependencies=dependencies,
        )


class Scheduler:
    satellite: Satellite
    world_query: WorldQuery
    pending_tasks: list[Task]
    running_tasks: list[Task]
    finished_tasks: set[Task]
    _current_energy_demand: u.Quantity

    def __init__(
        self,
        satellite: Satellite,
        world_query: WorldQuery,
    ):
        self.satellite = satellite
        self.world_query = world_query
        self.pending_tasks = []
        self._current_energy_demand = 0 << u.W * u.h  # type: ignore
        self._finished_tasks = set()

    def add_task(self, task: Task) -> None:
        self.pending_tasks.append(task)

    def _attempt_task(self, idx: int) -> bool:
        if not all(
            dep in self._finished_tasks for dep in self.pending_tasks[idx].dependencies
        ):  # dependency not met
            return False

        if self.satellite.battery.charge_level < (
            self.pending_tasks[idx].power * self.pending_tasks[idx].duration
            + self._current_energy_demand
        ):  # not enough energy
            return False

        self._current_energy_demand += (  # type: ignore
            self.pending_tasks[idx].power * self.pending_tasks[idx].duration
        )
        self.running_tasks.append(self.pending_tasks.pop(idx))
        return True

    def propagate(self, time: u.Quantity) -> None:
        for i in range(len(self.running_tasks)):
            time_decrement = min(self.running_tasks[i].duration, time)  # type: ignore
            self.running_tasks[i].duration -= time_decrement
            self._current_energy_demand -= (
                self.running_tasks[i].power * time_decrement  # type: ignore
            )

            if self.running_tasks[i].duration <= 1e-15 << u.minute:  # type: ignore
                self._finished_tasks.add(self.running_tasks.pop(i))

        for i in range(len(self.pending_tasks)):
            self._attempt_task(i)


class SimHandler:
    scheduler: Scheduler
    satellite: Satellite
    world_query: WorldQuery
    launch_time: datetime
    sim_stats: list[SimStatsPoint]

    def __init__(
        self,
        scheduler: Scheduler,
        satellite: Satellite,
        world_query: WorldQuery,
    ):
        self.scheduler = scheduler
        self.satellite = satellite
        self.world_query = world_query
        self.launch_time = world_query.launch_time
        self.sim_stats = []

    def calculate_power(self) -> u.Quantity:
        sky_field_t = self.world_query.ts.utc(
            self.world_query.current_time.year,
            self.world_query.current_time.month,
            self.world_query.current_time.day,
            self.world_query.current_time.hour,
            self.world_query.current_time.minute,
        )

        earth_pos = self.world_query.earth.at(sky_field_t)
        sun_pos = self.world_query.sun.at(sky_field_t)

        v_earth_to_sat = self.world_query.sat_orbit.r
        v_earth_to_sun = sun_pos.position.to(u.km) - earth_pos.position.to(u.km)
        v_sat_to_sun = sun_pos.position.to(u.km) - (
            self.world_query.sat_orbit.r + earth_pos.position.to(u.km)
        )

        cum_power = 0 << u.W  # type: ignore
        for panel in self.satellite.solar_panels:
            panel_direction = self.satellite.attitude.apply(panel.direction)
            sun_view_factor = max(dot_between(panel_direction, v_sat_to_sun), 0)
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
                * panel.surface_area
                * sun_view_factor
            )

        return cum_power

    def propagate(self, dt: u.Quantity) -> None:
        self.world_query.update_time(
            self.world_query.current_time
            + timedelta(minutes=dt.to(u.minute).to_value())  # type: ignore
        )
        self.satellite.propagate(dt)
        orbit_alignment = np.dot(
            self.satellite.attitude.as_matrix()[:, 2],
            self.world_query.sat_orbit.v / np.linalg.norm(self.world_query.sat_orbit.v),
        )

        power = self.calculate_power()
        self.satellite.battery.charge(power, dt)

        self.scheduler.propagate(dt)

        self.sim_stats.append(
            SimStatsPoint(
                time=self.world_query.current_time,
                power=power,
                orbit_alignment=orbit_alignment,
            )
        )


def main():
    world_query = WorldQuery()
    satellite: Satellite = Satellite.factory(world_query=world_query)
    scheduler: Scheduler = Scheduler(satellite=satellite, world_query=world_query)
    sim_handler: SimHandler = SimHandler(
        scheduler=scheduler,
        satellite=satellite,
        world_query=world_query,
    )

    for _ in tqdm(range(int(60 * 2 * 2))):
        sim_handler.propagate(5 << u.minute)  # type: ignore

    plt.plot([p.power.to_value() for p in sim_handler.sim_stats])
    # plt.plot([eh.orbit_alignment for eh in sim_handler.sim_stats])
    plt.show()


if __name__ == "__main__":
    main()
