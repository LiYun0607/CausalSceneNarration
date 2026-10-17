"""
TTC (Time-To-Collision) Baseline Safety Supervisor.

A reactive, physics-based safety baseline that uses Time-To-Collision
to trigger emergency braking. No semantic understanding of intent --
purely based on distance and relative velocity to nearby vehicles.

Used as a comparison baseline against the semantic SafetySupervisor.
"""

from __future__ import annotations

import logging
import math
from typing import Optional, Tuple

import numpy as np
import torch

try:
    import carla
except ImportError:
    carla = None

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider


class TTCSafetySupervisor:
    """
    Reactive TTC-based safety supervisor.

    Monitors Time-To-Collision with surrounding vehicles and applies
    emergency braking when TTC falls below threshold. Does NOT perform
    any semantic analysis (direction conflicts, junction awareness, etc.).
    """

    def __init__(
        self,
        vehicle: carla.Actor | None,
        world_map: carla.Map | None,
        global_plan=None,
        config: Optional[dict] = None,
    ):
        self.logger = logging.getLogger(__name__)
        self.vehicle = vehicle
        self.world_map = world_map
        self.global_plan = global_plan or []
        self.config = config or {}

        # TTC parameters
        self.ttc_threshold = self.config.get("ttc_threshold", 2.0)  # seconds
        self.ttc_brake_threshold = self.config.get("ttc_brake_threshold", 1.5)  # hard brake
        self.detection_range = self.config.get("detection_range", 30.0)  # meters
        self.detection_angle = self.config.get("detection_angle", 60.0)  # degrees (cone half-angle)

        # Control limits
        self.max_steer = self.config.get("max_steer_limit", 0.8)
        self.max_throttle = self.config.get("max_throttle_limit", 0.9)

        self.safety_enabled = True

        # Statistics
        self.total_checks = 0
        self.interventions = 0
        self.step = 0

        # Intervention logging
        self.intervention_log: list = []
        self.stuck_recovery_log: list = []

        # Stuck detection (same as semantic supervisor)
        self.stuck_speed_threshold = 0.1
        self.stuck_throttle_threshold = 0.2
        self.stuck_counter = 0
        self.stuck_threshold = 30
        self.last_position = None
        self.stuck_detected = False

        # TTC state
        self._emergency_brake_active = False
        self._emergency_brake_countdown = 0

        self.logger.info(
            "TTCSafetySupervisor initialized (ttc=%.1fs, range=%.0fm)",
            self.ttc_threshold, self.detection_range,
        )

    def enable_safety(self, enabled: bool = True):
        self.safety_enabled = enabled

    def set_safety_parameters(self, **kwargs):
        if "collision_time_threshold" in kwargs and kwargs["collision_time_threshold"] is not None:
            self.ttc_threshold = kwargs["collision_time_threshold"]
        if "max_steer_limit" in kwargs and kwargs["max_steer_limit"] is not None:
            self.max_steer = kwargs["max_steer_limit"]
        if "max_throttle_limit" in kwargs and kwargs["max_throttle_limit"] is not None:
            self.max_throttle = kwargs["max_throttle_limit"]

    def update_global_plan(self, global_plan):
        self.global_plan = global_plan or []

    def check_waypoint_safety(
        self,
        llm_waypoints,
        current_location,
        tick_data,
    ) -> Tuple[bool, torch.Tensor, float]:
        """
        TTC-based waypoint check.

        Unlike the semantic supervisor, this does NOT replace waypoints.
        It only flags unsafe situations for the control check to handle
        via emergency braking.
        """
        if not self.safety_enabled or llm_waypoints is None:
            return True, llm_waypoints, 0.0

        self.step = tick_data.get("step", self.step)
        self.total_checks += 1

        # Compute TTC
        ttc = self._compute_ttc()

        if ttc is not None and ttc < self.ttc_threshold:
            self._emergency_brake_active = True
            self._emergency_brake_countdown = 10
            self.interventions += 1
            self.intervention_log.append({
                "type": "ttc_brake", "step": self.step, "ttc": round(ttc, 3),
            })

            if self.step % 20 == 0 or ttc < self.ttc_brake_threshold:
                self.logger.warning(
                    "Step %d: TTC=%.2fs < %.2fs, emergency brake",
                    self.step, ttc, self.ttc_threshold,
                )
            return False, llm_waypoints, 0.0

        # Countdown for brake release
        if self._emergency_brake_countdown > 0:
            self._emergency_brake_countdown -= 1
            if self._emergency_brake_countdown == 0:
                self._emergency_brake_active = False

        return True, llm_waypoints, 0.0

    def check_control_safety(self, control: carla.VehicleControl) -> carla.VehicleControl:
        """Apply TTC-based emergency braking and control limits."""
        if not self.safety_enabled or not control:
            return control

        # Emergency braking override
        if self._emergency_brake_active:
            control.throttle = 0.0
            control.brake = 1.0
            return control

        # Normal control limits
        control.steer = float(np.clip(control.steer, -self.max_steer, self.max_steer))
        control.throttle = float(min(control.throttle, self.max_throttle))
        return control

    def detect_stuck(self, current_speed: float, current_throttle: float, current_location: np.ndarray) -> bool:
        """Detect if vehicle is stuck (same logic as semantic supervisor)."""
        if not self.safety_enabled:
            return False

        is_giving_throttle = current_throttle > self.stuck_throttle_threshold
        is_not_moving = current_speed < self.stuck_speed_threshold

        position_not_changed = False
        if self.last_position is not None:
            position_delta = np.linalg.norm(current_location - self.last_position)
            position_not_changed = position_delta < 0.2

        self.last_position = current_location.copy()

        if is_giving_throttle and is_not_moving and position_not_changed:
            self.stuck_counter += 1
            if self.stuck_counter >= self.stuck_threshold:
                if not self.stuck_detected:
                    self.logger.warning(
                        "Vehicle stuck: speed=%.2fm/s, throttle=%.2f, frames=%d",
                        current_speed, current_throttle, self.stuck_counter,
                    )
                    self.stuck_detected = True
                    self.intervention_log.append({"type": "stuck_detected", "step": self.step})
                    self.stuck_recovery_log.append({"start_step": self.step, "end_step": None, "recovered": False})
                return True
        else:
            if self.stuck_counter > 0:
                self.stuck_counter = 0
                if self.stuck_detected:
                    self.logger.info("Vehicle resumed moving")
                    if self.stuck_recovery_log and self.stuck_recovery_log[-1]["end_step"] is None:
                        self.stuck_recovery_log[-1]["end_step"] = self.step
                        self.stuck_recovery_log[-1]["recovered"] = True
                    self.stuck_detected = False

        return False

    def reset_stuck_detection(self):
        self.stuck_counter = 0
        self.stuck_detected = False
        self.last_position = None

    def get_intervention_summary(self) -> dict:
        """Return a summary dict suitable for JSON serialization."""
        ttc_brakes = [e for e in self.intervention_log if e["type"] == "ttc_brake"]
        stuck_events = [e for e in self.intervention_log if e["type"] == "stuck_detected"]
        return {
            "total_checks": self.total_checks,
            "total_interventions": self.interventions,
            "intervention_rate": self.interventions / max(1, self.total_checks),
            "ttc_brake_count": len(ttc_brakes),
            "stuck_detected_count": len(stuck_events),
            "intervention_log": self.intervention_log,
        }

    # ------------------------------------------------------------------ #
    # TTC computation
    # ------------------------------------------------------------------ #

    def _compute_ttc(self) -> Optional[float]:
        """
        Compute Time-To-Collision with the nearest vehicle in the forward cone.

        Returns TTC in seconds, or None if no collision risk detected.
        """
        if self.vehicle is None:
            return None

        try:
            ego_transform = self.vehicle.get_transform()
            ego_velocity = self.vehicle.get_velocity()
            ego_location = ego_transform.location
            ego_forward = ego_transform.get_forward_vector()

            ego_speed = math.sqrt(
                ego_velocity.x ** 2 + ego_velocity.y ** 2 + ego_velocity.z ** 2
            )

            # Skip if barely moving
            if ego_speed < 0.5:
                return None

            # Get all vehicles in the world
            world = CarlaDataProvider.get_world()
            if world is None:
                return None

            vehicles = world.get_actors().filter("vehicle.*")
            min_ttc = None

            for other in vehicles:
                if other.id == self.vehicle.id:
                    continue

                other_location = other.get_location()
                dx = other_location.x - ego_location.x
                dy = other_location.y - ego_location.y
                distance = math.sqrt(dx * dx + dy * dy)

                # Skip if too far
                if distance > self.detection_range or distance < 0.5:
                    continue

                # Check if in forward cone
                direction_to_other = carla.Vector3D(dx / distance, dy / distance, 0.0)
                dot = (
                    ego_forward.x * direction_to_other.x
                    + ego_forward.y * direction_to_other.y
                )
                angle = math.degrees(math.acos(max(-1.0, min(1.0, dot))))

                if angle > self.detection_angle:
                    continue

                # Compute relative velocity along the line connecting the two vehicles
                other_velocity = other.get_velocity()
                rel_vx = ego_velocity.x - other_velocity.x
                rel_vy = ego_velocity.y - other_velocity.y

                # Project relative velocity onto the direction to other vehicle
                closing_speed = (
                    rel_vx * direction_to_other.x + rel_vy * direction_to_other.y
                )

                # Only consider if we are closing in
                if closing_speed <= 0.1:
                    continue

                # Account for vehicle dimensions (approximate bounding box)
                effective_distance = max(0.1, distance - 4.0)  # ~4m for two car lengths
                ttc = effective_distance / closing_speed

                if min_ttc is None or ttc < min_ttc:
                    min_ttc = ttc

            return min_ttc

        except Exception as e:
            self.logger.debug("TTC computation error: %s", e)
            return None
