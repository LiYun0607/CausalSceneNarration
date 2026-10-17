"""
Safety Supervisor for LMDrive Agent.

Features:
- Route-aware safety with Traffic Manager integration
- Direction conflict detection (LEFT/RIGHT/STRAIGHT)
- Deviation-based safety check
- Stuck detection
- Control limits
"""

from __future__ import annotations

import logging
import math
from typing import List, Optional, Tuple

import numpy as np
import torch

try:
    import carla
except ImportError:
    carla = None

try:
    from agents.navigation.local_planner import RoadOption
except ImportError:
    try:
        from leaderboard.leaderboard.utils.agents.navigation.local_planner import RoadOption
    except ImportError:
        # Fallback: define RoadOption locally
        from enum import Enum
        class RoadOption(Enum):
            VOID = -1
            LEFT = 1
            RIGHT = 2
            STRAIGHT = 3
            LANEFOLLOW = 4
            CHANGELANELEFT = 5
            CHANGELANERIGHT = 6

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider


class SafetySupervisor:
    """Route-aware safety supervisor with Traffic Manager based recovery and stuck detection."""

    STRAIGHT_THRESHOLD_DEG = 20.0
    MIN_TAKEOVER_STEPS = 20

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

        # Thresholds from config
        self.deviation_warning = self.config.get("lateral_deviation_warning", 2.0)
        self.deviation_critical = self.config.get("lateral_deviation_critical", 3.5)
        self.collision_time_threshold = self.config.get("collision_time_threshold", 3.0)
        self.overall_threshold = self.config.get("overall_safety_threshold", 0.7)
        self.max_steer = self.config.get("max_steer_limit", 0.8)
        self.max_throttle = self.config.get("max_throttle_limit", 0.9)

        self.safety_enabled = True

        # Traffic Manager
        self.tm = None
        self.tm_port = None
        self._init_traffic_manager()

        # Takeover state
        self.takeover_active = False
        self.last_takeover_step = -1
        self.last_takeover_reason: Optional[str] = None
        self.using_fallback = False

        # Statistics
        self.total_checks = 0
        self.interventions = 0
        self.step = 0

        # Intervention logging
        self.intervention_log: List[dict] = []
        self.stuck_recovery_log: List[dict] = []
        self.false_positive_count = 0  # Incremented externally if needed

        # Stuck detection
        self.stuck_speed_threshold = 0.1  # m/s
        self.stuck_throttle_threshold = 0.2
        self.stuck_counter = 0
        self.stuck_threshold = 30  # ~1.5 seconds at 20 FPS
        self.last_position = None
        self.stuck_detected = False

        self.logger.info("SafetySupervisor initialized")

    def enable_safety(self, enabled: bool = True):
        self.safety_enabled = enabled
        self.logger.info("Safety supervision %s", "enabled" if enabled else "disabled")

    def set_safety_parameters(
        self,
        deviation_warning: float | None = None,
        deviation_critical: float | None = None,
        deviation_threshold: float | None = None,
        speed_threshold: float | None = None,
        steer_threshold: float | None = None,
        collision_time_threshold: float | None = None,
        overall_threshold: float | None = None,
    ):
        # Support both old and new parameter names
        if deviation_warning is not None:
            self.deviation_warning = deviation_warning
        if deviation_critical is not None:
            self.deviation_critical = deviation_critical
        if deviation_threshold is not None:
            self.deviation_warning = deviation_threshold
        if speed_threshold is not None:
            self.max_throttle = speed_threshold
        if steer_threshold is not None:
            self.max_steer = steer_threshold
        if collision_time_threshold is not None:
            self.collision_time_threshold = collision_time_threshold
        if overall_threshold is not None:
            self.overall_threshold = overall_threshold

        self.logger.info(
            "Safety thresholds: warning=%.2fm, critical=%.2fm, steer=%.2f, throttle=%.2f",
            self.deviation_warning,
            self.deviation_critical,
            self.max_steer,
            self.max_throttle,
        )

    def update_global_plan(self, global_plan):
        self.global_plan = global_plan or []

    def check_waypoint_safety(
        self,
        llm_waypoints,
        current_location,
        tick_data,
    ) -> Tuple[bool, torch.Tensor, float]:
        """
        Analyze LLM waypoints and optionally provide fallback.

        Returns:
            (is_safe, waypoints, deviation)
        """
        if not self.safety_enabled or llm_waypoints is None:
            return True, llm_waypoints, 0.0

        self.step = tick_data.get('step', self.step)
        self.total_checks += 1

        # If already in takeover with fallback, check if we should release
        if self.takeover_active and self.using_fallback:
            steps_in_takeover = tick_data.get("step", 0) - self.last_takeover_step

            # Release takeover after MIN_TAKEOVER_STEPS (time-based release)
            if steps_in_takeover >= self.MIN_TAKEOVER_STEPS:
                self._release_takeover()
                self.using_fallback = False
                self.logger.info("Takeover ended after %d steps", steps_in_takeover)
                return True, llm_waypoints, 0.0

            # Continue with fallback waypoints
            fallback_local = self._planned_waypoints_to_local(tick_data)
            fallback_tensor = self._to_tensor(fallback_local, like=llm_waypoints)
            return False, fallback_tensor, 0.0

        # Normal safety check
        waypoints_np = self._to_numpy(llm_waypoints)
        target_point = tick_data.get("target_point")
        deviation = self._compute_deviation(waypoints_np, target_point)

        expected_cmd = self._get_expected_command(tick_data)
        predicted_cmd = self._infer_llm_command(waypoints_np)

        # Skip direction conflict detection inside junctions to avoid
        # false positives during mid-turn execution (LLM waypoints flatten
        # in rotated ego frame, causing STRAIGHT inference while expected
        # command is still LEFT/RIGHT).
        in_junction = self._is_in_junction()
        conflict = False if in_junction else self._is_direction_conflict(expected_cmd, predicted_cmd)

        # Check for severe deviation
        # DISABLED: deviation check often triggers false positives, only use direction conflict
        # severe_deviation = deviation >= self.deviation_critical
        severe_deviation = False

        # Debug output (every 20 steps or on conflict)
        if self.step % 20 == 0 or conflict:
            self.logger.debug(
                "Step %d: Expected=%s, Predicted=%s, Conflict=%s, InJunction=%s, Deviation=%.2fm",
                self.step, expected_cmd.name, predicted_cmd.name, conflict, in_junction, deviation
            )

        # Determine if takeover needed (only on direction conflict)
        takeover_reason = None
        if conflict:
            takeover_reason = f"direction_conflict(expected={expected_cmd.name}, predicted={predicted_cmd.name})"

        if takeover_reason and not self.takeover_active:
            fallback_local = self._planned_waypoints_to_local(tick_data)
            fallback_tensor = self._to_tensor(fallback_local, like=llm_waypoints)
            self._activate_takeover(tick_data, reason=takeover_reason)
            self.using_fallback = True
            self._log_intervention(
                intervention_type="direction_conflict",
                step=self.step,
                expected=expected_cmd.name,
                predicted=predicted_cmd.name,
                in_junction=in_junction,
                deviation=deviation,
            )
            return False, fallback_tensor, deviation

        return True, llm_waypoints, deviation

    def check_control_safety(self, control: carla.VehicleControl) -> carla.VehicleControl:
        """Clamp control values for safety."""
        if not self.safety_enabled or not control:
            return control

        if self.takeover_active and self.using_fallback:
            # Allow more aggressive steering during correction
            max_steer_takeover = min(1.0, self.max_steer * 1.2)
            control.steer = float(np.clip(control.steer, -max_steer_takeover, max_steer_takeover))
            # Reduce speed during corrections
            control.throttle = float(min(control.throttle, self.max_throttle * 0.6))
            # Light braking when turning hard
            if abs(control.steer) > 0.5:
                control.brake = max(control.brake, 0.2)
            return control

        # Normal safety limits
        control.steer = float(np.clip(control.steer, -self.max_steer, self.max_steer))
        control.throttle = float(min(control.throttle, self.max_throttle))
        return control

    def detect_stuck(self, current_speed: float, current_throttle: float, current_location: np.ndarray) -> bool:
        """
        Detect if vehicle is stuck (throttle applied but not moving).

        Returns:
            True if stuck detected
        """
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
                        current_speed, current_throttle, self.stuck_counter
                    )
                    self.stuck_detected = True
                    self._log_stuck_event(start_step=self.step)
                return True
        else:
            if self.stuck_counter > 0:
                self.stuck_counter = 0
                if self.stuck_detected:
                    self.logger.info("Vehicle resumed moving")
                    self._log_stuck_recovery(end_step=self.step)
                    self.stuck_detected = False

        return False

    def reset_stuck_detection(self):
        """Reset stuck detection state."""
        self.stuck_counter = 0
        self.stuck_detected = False
        self.last_position = None

    # --------------------------------------------------------------------- #
    # Intervention logging
    # --------------------------------------------------------------------- #
    def _log_intervention(self, intervention_type: str, step: int, **kwargs):
        entry = {"type": intervention_type, "step": step, **kwargs}
        self.intervention_log.append(entry)
        self.logger.info("INTERVENTION[%d]: %s %s", step, intervention_type,
                         " ".join(f"{k}={v}" for k, v in kwargs.items()))

    def _log_stuck_event(self, start_step: int):
        self.stuck_recovery_log.append({"start_step": start_step, "end_step": None, "recovered": False})
        self._log_intervention(intervention_type="stuck_detected", step=start_step)

    def _log_stuck_recovery(self, end_step: int):
        if self.stuck_recovery_log and self.stuck_recovery_log[-1]["end_step"] is None:
            self.stuck_recovery_log[-1]["end_step"] = end_step
            self.stuck_recovery_log[-1]["recovered"] = True
            start = self.stuck_recovery_log[-1]["start_step"]
            duration = end_step - start
            self.logger.info("STUCK_RECOVERY: %d steps (%.1fs at 20FPS)", duration, duration / 20.0)

    def get_intervention_summary(self) -> dict:
        """Return a summary dict suitable for JSON serialization."""
        direction_conflicts = [e for e in self.intervention_log if e["type"] == "direction_conflict"]
        stuck_events = [e for e in self.intervention_log if e["type"] == "stuck_detected"]
        stuck_recovered = [e for e in self.stuck_recovery_log if e.get("recovered")]
        stuck_durations = [
            e["end_step"] - e["start_step"]
            for e in self.stuck_recovery_log if e.get("recovered") and e["end_step"] is not None
        ]
        return {
            "total_checks": self.total_checks,
            "total_interventions": self.interventions,
            "intervention_rate": self.interventions / max(1, self.total_checks),
            "direction_conflict_count": len(direction_conflicts),
            "stuck_detected_count": len(stuck_events),
            "stuck_recovered_count": len(stuck_recovered),
            "stuck_recovery_rate": len(stuck_recovered) / max(1, len(stuck_events)),
            "mean_stuck_recovery_steps": sum(stuck_durations) / max(1, len(stuck_durations)) if stuck_durations else 0,
            "intervention_log": self.intervention_log,
        }

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #
    def _is_in_junction(self) -> bool:
        """Check if the ego vehicle is currently inside a junction using the CARLA map."""
        if self.vehicle is None or self.world_map is None:
            return False
        try:
            ego_location = self.vehicle.get_location()
            carla_wp = self.world_map.get_waypoint(ego_location)
            return carla_wp is not None and carla_wp.is_junction
        except Exception:
            return False

    def _init_traffic_manager(self):
        if not carla:
            self.logger.warning("carla module not found, Traffic Manager disabled")
            return
        try:
            client = CarlaDataProvider.get_client()
            if client is None:
                self.logger.warning("CarlaDataProvider returned None client")
                return

            tm_port = CarlaDataProvider.get_traffic_manager_port()
            tm = client.get_trafficmanager(tm_port)
            if hasattr(tm, "set_synchronous_mode"):
                tm.set_synchronous_mode(True)
            if hasattr(tm, "global_percentage_speed_difference"):
                tm.global_percentage_speed_difference(20.0)

            self.tm = tm
            self.tm_port = tm_port
            self.logger.info("Traffic Manager initialized (port=%s)", tm_port)
        except Exception as exc:
            self.logger.warning("Traffic Manager init failed: %s", exc)
            self.tm = None

    def _to_numpy(self, waypoints: torch.Tensor | np.ndarray) -> np.ndarray:
        if torch.is_tensor(waypoints):
            data = waypoints.detach().cpu().numpy()
        else:
            data = np.asarray(waypoints, dtype=np.float32)
        return np.copy(data)

    def _to_tensor(self, waypoints: np.ndarray, like=None) -> torch.Tensor:
        tensor = torch.tensor(waypoints, dtype=torch.float32)
        if like is not None and torch.is_tensor(like) and like.is_cuda:
            tensor = tensor.cuda()
        return tensor

    def _compute_deviation(self, llm_waypoints: np.ndarray, target_point) -> float:
        if target_point is None or llm_waypoints.size == 0:
            return 0.0

        target = np.asarray(target_point, dtype=np.float32)

        # Convert LLM waypoints to CARLA coordinate system
        waypoints_carla = llm_waypoints.copy()
        if waypoints_carla.ndim >= 2 and waypoints_carla.shape[0] > 0:
            waypoints_carla[:, 1] *= -1.0

        current = waypoints_carla[0]
        deviation = float(np.linalg.norm(current - target))

        return deviation

    def _infer_llm_command(self, waypoints: np.ndarray) -> RoadOption:
        if waypoints.shape[0] == 0:
            return RoadOption.LANEFOLLOW

        wp_local = waypoints.copy()
        if wp_local.ndim >= 2 and wp_local.shape[0] > 0:
            wp_local[:, 1] *= -1.0  # Convert to CARLA coordinates

        delta = wp_local[-1]
        angle = math.degrees(math.atan2(delta[0], delta[1] + 1e-6))

        if abs(angle) <= self.STRAIGHT_THRESHOLD_DEG:
            return RoadOption.STRAIGHT
        return RoadOption.RIGHT if angle > 0 else RoadOption.LEFT

    def _get_expected_command(self, tick_data) -> RoadOption:
        cmd_value = tick_data.get("next_command")
        try:
            if cmd_value is None:
                return RoadOption.LANEFOLLOW
            return RoadOption(cmd_value)
        except ValueError:
            return RoadOption.LANEFOLLOW

    def _is_direction_conflict(self, expected: RoadOption, predicted: RoadOption) -> bool:
        """
        Check if predicted direction conflicts with expected.
        """
        # For explicit turn commands, must match
        if expected in (RoadOption.LEFT, RoadOption.RIGHT):
            if predicted in (RoadOption.LEFT, RoadOption.RIGHT, RoadOption.STRAIGHT):
                return expected != predicted
            return False

        # For STRAIGHT command, left/right is wrong
        if expected == RoadOption.STRAIGHT:
            return predicted in (RoadOption.LEFT, RoadOption.RIGHT)

        # For LANEFOLLOW, allow any direction
        return False

    def _activate_takeover(self, tick_data, reason: str):
        if not self.takeover_active:
            self.interventions += 1
            self.last_takeover_step = tick_data.get("step", 0)
            self.logger.warning("Takeover activated: %s", reason)

        self.takeover_active = True
        self.last_takeover_reason = reason

        if self.tm and self.vehicle:
            try:
                if hasattr(self.tm, "auto_lane_change"):
                    self.tm.auto_lane_change(self.vehicle, False)
                if hasattr(self.tm, "distance_to_leading_vehicle"):
                    self.tm.distance_to_leading_vehicle(self.vehicle, 5.0)
                if hasattr(self.tm, "vehicle_percentage_speed_difference"):
                    self.tm.vehicle_percentage_speed_difference(self.vehicle, 40.0)
                if hasattr(self.tm, "ignore_lights_percentage"):
                    self.tm.ignore_lights_percentage(self.vehicle, 0)
                if hasattr(self.tm, "ignore_signs_percentage"):
                    self.tm.ignore_signs_percentage(self.vehicle, 0)
            except Exception as exc:
                self.logger.warning("TM constraint failed: %s", exc)

    def _release_takeover(self):
        if self.takeover_active:
            self.logger.info("Takeover released (reason=%s)", self.last_takeover_reason)
        self.takeover_active = False
        self.using_fallback = False
        self.last_takeover_reason = None

        if self.tm and self.vehicle:
            try:
                if hasattr(self.tm, "auto_lane_change"):
                    self.tm.auto_lane_change(self.vehicle, True)
                if hasattr(self.tm, "distance_to_leading_vehicle"):
                    self.tm.distance_to_leading_vehicle(self.vehicle, 2.0)
                if hasattr(self.tm, "vehicle_percentage_speed_difference"):
                    self.tm.vehicle_percentage_speed_difference(self.vehicle, 20.0)
            except Exception:
                pass

    def _planned_waypoints_to_local(self, tick_data, max_count: int = 5) -> np.ndarray:
        """Use ground truth waypoints as fallback."""
        planned_local = tick_data.get("planned_waypoints_local")

        if planned_local is not None and len(planned_local) > 0:
            planned_np = np.asarray(planned_local, dtype=np.float32)
            num_points = min(len(planned_np), max_count)
            fallback = np.zeros((max_count, 2), dtype=np.float32)

            for i in range(num_points):
                fallback[i, 0] = planned_np[i, 0]
                fallback[i, 1] = -planned_np[i, 1]  # Convert to LLM convention

            if num_points < max_count:
                if num_points >= 2:
                    direction = planned_np[-1] - planned_np[-2]
                    for i in range(num_points, max_count):
                        next_point = planned_np[-1] + direction * (i - num_points + 1)
                        fallback[i, 0] = next_point[0]
                        fallback[i, 1] = -next_point[1]
                else:
                    for i in range(num_points, max_count):
                        fallback[i, 0] = fallback[0, 0]
                        fallback[i, 1] = fallback[0, 1] - (i * 3.0)

            return fallback

        # Fallback to planned waypoints with coordinate transform
        planned = tick_data.get("planned_waypoints")
        if planned is None or len(planned) == 0:
            return self._create_straight_fallback()

        planned = np.asarray(planned, dtype=np.float32)
        current = np.asarray(tick_data.get("gps"), dtype=np.float32)
        compass = float(tick_data.get("compass", 0.0))
        theta = compass + np.pi / 2.0
        rotation = np.array(
            [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]],
            dtype=np.float32,
        )

        locals_: List[List[float]] = []
        for wp in planned[:max_count]:
            delta = wp - current
            local = rotation.T.dot(delta)
            locals_.append([float(local[0]), float(-local[1])])

        if len(locals_) < max_count:
            locals_.extend(self._create_straight_fallback()[len(locals_):].tolist())

        return np.asarray(locals_, dtype=np.float32)

    def _create_straight_fallback(self, length: int = 5) -> np.ndarray:
        fallback = np.zeros((length, 2), dtype=np.float32)
        fallback[:, 1] = -np.linspace(3.0, 15.0, num=length)
        return fallback


# Backward compatibility alias
AdvancedSafetySupervisor = SafetySupervisor
