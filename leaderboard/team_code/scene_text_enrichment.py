"""
Causal Scene Narration (CSN) for LMDrive — Dual-Channel Architecture.

Restructures LMDrive's two text channels into a unified causal reasoning system:

  INSTRUCTION channel (64 tokens):
    Intent-Constraint Causal Alignment — navigation intent fused with ranked
    environmental constraints through causal connectors (BUT, YIELD BEFORE,
    BECAUSE).  Example:
      "Turn left at intersection, BUT yield to pedestrian crossing from right
       12m. Red light, stop before turn."

  NOTICE channel (64 tokens):
    Scene Graph Serialization — structured quantitative environment data with
    physical units.  Example:
      "[EGO] 32/50km/h. [ROAD] Junction, 2-lane, wet. [ACTORS] Sedan ahead
       25m 40km/h."

Core principles:
  1. Intent-Constraint Causal Alignment: constraints reference navigation intent
     through explicit causal connectors.
  2. Quantitative Physical Grounding: meters, km/h, m/s — no vague qualifiers.
  3. Structured Information Separation: instruction carries causal reasoning,
     notice carries quantitative scene context.

Zero VRAM cost — all CPU computation via CARLA 0.9.10 Python API.
"""

import logging
import math
import numpy as np

try:
    import carla
except ImportError:
    carla = None

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Navigation command mapping (RoadOption enum values)
# ---------------------------------------------------------------------------
_NAV_COMMANDS = {
    1: "turn left",
    2: "turn right",
    3: "go straight",
    4: "follow lane",
    5: "change lane left",
    6: "change lane right",
}

# Compact intent phrases for the instruction channel.
_INTENT_PHRASES = {
    1: "Turn left",
    2: "Turn right",
    3: "Go straight",
    4: "Follow lane",
    5: "Change to left lane",
    6: "Change to right lane",
}

# Which side of the ego vehicle is "conflict side" for each manoeuvre.
_CONFLICT_SIDES = {
    1: {"ahead-left", "ahead", "left"},
    2: {"ahead-right", "ahead", "right"},
    3: {"ahead"},
    4: {"ahead"},
    5: {"ahead-left", "left", "ahead"},
    6: {"ahead-right", "right", "ahead"},
}

# Token budget (leave margin below the 64-token hard limit)
_MAX_TOKENS = 58


# ===================================================================
# Main class
# ===================================================================
class SceneTextEnrichment:
    """Dual-channel Causal Scene Narration from CARLA APIs."""

    def __init__(self, vehicle, world, world_map, config=None):
        self._vehicle = vehicle
        self._world = world
        self._map = world_map

        cfg = config or {}
        self._vehicle_radius = cfg.get("vehicle_radius", 50.0)
        self._ped_radius = cfg.get("pedestrian_radius", 30.0)
        self._max_vehicles = cfg.get("max_vehicles_reported", 3)
        self._max_pedestrians = cfg.get("max_pedestrians_reported", 2)

        import os
        self._flat_mode = os.environ.get(
            "LMDRIVE_TEXT_FLAT", "false"
        ).lower() in ("true", "1", "yes")

        # Perception noise injection for ablation
        self._noise_dist_std = float(os.environ.get("CSN_NOISE_DIST_STD", "0"))
        self._noise_speed_rel = float(os.environ.get("CSN_NOISE_SPEED_REL", "0"))
        self._noise_miss_rate = float(os.environ.get("CSN_NOISE_MISS_RATE", "0"))
        self._noise_rng = np.random.default_rng(42) if (
            self._noise_dist_std > 0 or self._noise_speed_rel > 0
            or self._noise_miss_rate > 0) else None

        mode_str = "FLAT (no causal)" if self._flat_mode else "CAUSAL"
        noise_str = ""
        if self._noise_rng is not None:
            noise_str = (f", noise: dist_std={self._noise_dist_std}m, "
                        f"speed_rel={self._noise_speed_rel}, "
                        f"miss_rate={self._noise_miss_rate}")
        logger.info(
            "CSN dual-channel initialised (veh_r=%.0f, ped_r=%.0f, mode=%s%s)",
            self._vehicle_radius, self._ped_radius, mode_str, noise_str,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, nav_command, base_instruction="", scenario_notice="",
                 tick_data=None):
        """Primary entry point — returns (instruction, notice) tuple.

        Args:
            nav_command:      int (1-6) from RoadOption enum.
            base_instruction: Original template instruction from planner.
            scenario_notice:  Original scenario notice from pos2notice().
            tick_data:        Agent tick data dict (for next_command fallback).

        Returns:
            (causal_instruction, scene_notice) — two strings, each ≤58 tokens.
            Falls back to (base_instruction, scenario_notice) on error.
        """
        try:
            state = self._extract_scene_state()
            constraints = self._identify_causal_constraints(nav_command, state)
            if self._flat_mode:
                instruction = self._render_instruction_flat(
                    nav_command, constraints, scenario_notice
                )
            else:
                instruction = self._render_instruction(
                    nav_command, constraints, scenario_notice
                )
            notice = self._render_notice(state)
            return instruction, notice
        except Exception:
            logger.debug("CSN generate failed, falling back", exc_info=True)
            return base_instruction, scenario_notice

    def generate_scene_text(self, tick_data=None):
        """Backward-compatible single-string wrapper (notice channel only).

        Used when text enrichment only replaces the notice channel.
        """
        nav_command = 4
        if tick_data is not None and "next_command" in tick_data:
            nav_command = int(tick_data["next_command"])
        try:
            state = self._extract_scene_state()
            constraints = self._identify_causal_constraints(nav_command, state)
            return self._render_single(nav_command, state, constraints)
        except Exception:
            logger.debug("CSN single-channel fallback failed", exc_info=True)
            return ""

    # ------------------------------------------------------------------
    # Scene state extraction  (CARLA 0.9.10 API, CPU only)
    # ------------------------------------------------------------------

    def _extract_scene_state(self):
        ego_tf = self._vehicle.get_transform()
        ego_loc = ego_tf.location
        ego_vel = self._vehicle.get_velocity()
        ego_speed_ms = math.sqrt(ego_vel.x**2 + ego_vel.y**2 + ego_vel.z**2)
        fwd = ego_tf.get_forward_vector()
        right = ego_tf.get_right_vector()

        return {
            "ego_speed_kmh": ego_speed_ms * 3.6,
            "ego_location": ego_loc,
            "ego_fwd": fwd,
            "ego_right": right,
            "traffic_light": self._read_traffic_light(ego_loc),
            "speed_limit": self._read_speed_limit(),
            "vehicles": self._read_vehicles(ego_loc, fwd, right),
            "pedestrians": self._read_pedestrians(ego_loc, fwd, right),
            "road": self._read_road(ego_loc),
            "weather": self._read_weather(),
        }

    def _read_traffic_light(self, ego_loc):
        try:
            if self._vehicle.is_at_traffic_light():
                light = self._vehicle.get_traffic_light()
                if light is not None:
                    s = _tl_state_str(self._vehicle.get_traffic_light_state())
                    if s:
                        d = ego_loc.distance(light.get_transform().location)
                        return {"state": s, "distance_m": d}
        except Exception:
            pass
        try:
            lights = self._world.get_actors().filter("traffic.traffic_light")
            best = None
            for tl in lights:
                d = ego_loc.distance(tl.get_transform().location)
                if d > 50.0:
                    continue
                s = _tl_state_str(tl.state)
                if not s:
                    continue
                if best is None or d < best["distance_m"]:
                    best = {"state": s, "distance_m": d}
            return best
        except Exception:
            return None

    def _read_speed_limit(self):
        try:
            sl = self._vehicle.get_speed_limit()
            if sl is not None and sl > 0:
                return float(sl)
        except Exception:
            pass
        return None

    def _read_vehicles(self, ego_loc, fwd, right):
        try:
            actors = self._world.get_actors().filter("vehicle.*")
        except Exception:
            return []

        ego_speed_ms = math.sqrt(
            self._vehicle.get_velocity().x**2
            + self._vehicle.get_velocity().y**2
        )
        nearby = []
        for v in actors:
            if v.id == self._vehicle.id:
                continue
            v_loc = v.get_location()
            dist = ego_loc.distance(v_loc)
            if dist > self._vehicle_radius:
                continue
            dx = v_loc.x - ego_loc.x
            dy = v_loc.y - ego_loc.y
            dot_fwd = dx * fwd.x + dy * fwd.y
            dot_right = dx * right.x + dy * right.y
            if dot_fwd < -5.0:
                continue
            v_vel = v.get_velocity()
            v_speed_ms = math.sqrt(v_vel.x**2 + v_vel.y**2)
            v_fwd_component = v_vel.x * fwd.x + v_vel.y * fwd.y
            nearby.append({
                "distance_m": dist,
                "position": _position_label(dot_fwd, dot_right),
                "speed_kmh": v_speed_ms * 3.6,
                "rel_speed_kmh": (ego_speed_ms - v_fwd_component) * 3.6,
                "dot_fwd": dot_fwd,
                "dot_right": dot_right,
            })
        nearby.sort(key=lambda v: v["distance_m"])
        # Apply perception noise
        if self._noise_rng is not None:
            # Miss detections
            if self._noise_miss_rate > 0:
                nearby = [v for v in nearby
                          if self._noise_rng.random() > self._noise_miss_rate]
            for v in nearby:
                if self._noise_dist_std > 0:
                    v["distance_m"] = max(1.0,
                        v["distance_m"] + self._noise_rng.normal(0, self._noise_dist_std))
                if self._noise_speed_rel > 0:
                    noise_factor = 1.0 + self._noise_rng.normal(0, self._noise_speed_rel)
                    v["speed_kmh"] = max(0, v["speed_kmh"] * noise_factor)
                    v["rel_speed_kmh"] = v["rel_speed_kmh"] * noise_factor
        return nearby

    def _read_pedestrians(self, ego_loc, fwd, right):
        try:
            actors = self._world.get_actors().filter("walker.pedestrian.*")
        except Exception:
            return []

        nearby = []
        for w in actors:
            w_loc = w.get_location()
            dist = ego_loc.distance(w_loc)
            if dist > self._ped_radius:
                continue
            dx = w_loc.x - ego_loc.x
            dy = w_loc.y - ego_loc.y
            dot_fwd = dx * fwd.x + dy * fwd.y
            dot_right = dx * right.x + dy * right.y
            if dot_fwd < -3.0:
                continue
            w_vel = w.get_velocity()
            w_speed = math.sqrt(w_vel.x**2 + w_vel.y**2)
            w_lateral = w_vel.x * right.x + w_vel.y * right.y
            if w_speed > 0.3:
                if w_lateral > 0.3:
                    crossing = "crossing right-to-left"
                elif w_lateral < -0.3:
                    crossing = "crossing left-to-right"
                else:
                    crossing = "walking along road"
            else:
                crossing = "stationary"
            nearby.append({
                "distance_m": dist,
                "position": _position_label(dot_fwd, dot_right),
                "speed_ms": w_speed,
                "crossing_dir": crossing,
                "dot_fwd": dot_fwd,
                "dot_right": dot_right,
            })
        nearby.sort(key=lambda p: p["distance_m"])
        # Apply perception noise
        if self._noise_rng is not None:
            if self._noise_miss_rate > 0:
                nearby = [p for p in nearby
                          if self._noise_rng.random() > self._noise_miss_rate]
            for p in nearby:
                if self._noise_dist_std > 0:
                    p["distance_m"] = max(1.0,
                        p["distance_m"] + self._noise_rng.normal(0, self._noise_dist_std))
                if self._noise_speed_rel > 0:
                    p["speed_ms"] = max(0,
                        p["speed_ms"] * (1.0 + self._noise_rng.normal(0, self._noise_speed_rel)))
        return nearby

    def _read_road(self, ego_loc):
        info = {"in_junction": False, "junction_ahead": False, "lane_count": 1}
        try:
            wp = self._map.get_waypoint(ego_loc)
            if wp is None:
                return info
            info["in_junction"] = wp.is_junction
            if not wp.is_junction:
                try:
                    nexts = wp.next(20.0)
                    if nexts and any(w.is_junction for w in nexts):
                        info["junction_ahead"] = True
                except Exception:
                    pass
            info["lane_count"] = (
                1 + _count_lanes(wp, "left") + _count_lanes(wp, "right")
            )
        except Exception:
            pass
        return info

    def _read_weather(self):
        conditions = []
        try:
            w = self._world.get_weather()
            if w.precipitation > 40:
                conditions.append("heavy rain")
            elif w.precipitation > 15:
                conditions.append("rain")
            if w.fog_density > 40:
                conditions.append("dense fog")
            elif w.fog_density > 15:
                conditions.append("fog")
            if w.wetness > 50:
                conditions.append("wet road")
            if w.sun_altitude_angle < -5:
                conditions.append("night")
        except Exception:
            pass
        return conditions

    # ------------------------------------------------------------------
    # Causal constraint identification
    # ------------------------------------------------------------------

    def _identify_causal_constraints(self, nav_command, state):
        """Score and rank constraints by causal relevance to navigation intent."""
        intent = _NAV_COMMANDS.get(nav_command, "follow lane")
        conflict_sides = _CONFLICT_SIDES.get(nav_command, {"ahead"})
        constraints = []

        # --- Traffic light ---
        tl = state["traffic_light"]
        if tl is not None:
            dist = tl["distance_m"]
            tl_state = tl["state"]
            if tl_state == "Red":
                causal = f"BUT red light {dist:.0f}m ahead, stop BEFORE {intent}"
                priority = 100.0 + max(0, 50.0 - dist)
            elif tl_state == "Yellow":
                causal = f"BUT yellow light {dist:.0f}m, prepare to stop"
                priority = 80.0 + max(0, 50.0 - dist)
            elif dist < 10:
                causal = f"green light, proceed to {intent}"
                priority = 10.0
            else:
                causal = None
                priority = 0.0

            if causal:
                constraints.append({
                    "kind": "traffic_light",
                    "priority": priority,
                    "causal": causal,
                    "text_short": f"{tl_state} {dist:.0f}m",
                })

        # --- Vehicles ---
        for veh in state["vehicles"][:self._max_vehicles]:
            dist = veh["distance_m"]
            pos = veh["position"]
            spd = veh["speed_kmh"]
            rel_spd = veh["rel_speed_kmh"]
            on_conflict = pos in conflict_sides
            is_stopped = spd < 2.0
            is_approaching = rel_spd > 5.0

            base = max(0, 50.0 - dist)
            if on_conflict:
                base *= 2.5
            else:
                base *= 0.4
            if is_stopped and "ahead" in pos:
                base *= 2.0
            if is_approaching and on_conflict:
                base *= 1.5

            causal = _vehicle_causal(
                intent, nav_command, pos, is_stopped, is_approaching, dist
            )
            spd_str = "stopped" if is_stopped else f"{spd:.0f}km/h"
            constraints.append({
                "kind": "vehicle",
                "priority": base,
                "causal": causal,
                "text_short": f"veh {pos} {dist:.0f}m {spd_str}",
            })

        # --- Pedestrians ---
        for ped in state["pedestrians"][:self._max_pedestrians]:
            dist = ped["distance_m"]
            pos = ped["position"]
            spd = ped["speed_ms"]
            crossing = ped["crossing_dir"]
            on_conflict = pos in conflict_sides
            is_crossing = "crossing" in crossing

            base = max(0, 40.0 - dist)
            if is_crossing:
                base *= 3.0
            if on_conflict:
                base *= 2.0
            else:
                base *= 0.5

            causal = _pedestrian_causal(intent, pos, is_crossing, crossing, dist)
            cross_str = crossing if is_crossing else "stationary"
            constraints.append({
                "kind": "pedestrian",
                "priority": base,
                "causal": causal,
                "text_short": f"ped {pos} {dist:.0f}m {cross_str}",
            })

        # --- Speed violation ---
        sl = state["speed_limit"]
        ego_spd = state["ego_speed_kmh"]
        if sl is not None and ego_spd - sl > 10:
            constraints.append({
                "kind": "speed",
                "priority": 60.0,
                "causal": f"BECAUSE over speed limit, slow down BEFORE {intent}",
                "text_short": f"{ego_spd:.0f}/{sl:.0f}km/h over",
            })

        constraints.sort(key=lambda c: c["priority"], reverse=True)
        return constraints

    # ------------------------------------------------------------------
    # INSTRUCTION channel: Intent + Causal Constraints
    # ------------------------------------------------------------------

    def _render_instruction(self, nav_command, constraints, scenario_notice=""):
        """Render causal instruction: intent fused with top constraints.

        Format: "<Intent phrase>, BUT <constraint1>. <constraint2>."
        Budget: ~58 tokens.
        """
        intent_phrase = _INTENT_PHRASES.get(nav_command, "Follow lane")

        # Collect causal fragments that fit within token budget
        # Intent phrase ≈ 3-5 tokens
        tokens_used = _estimate_tokens(intent_phrase)
        causal_parts = []

        for c in constraints:
            if c["priority"] <= 0:
                continue
            frag = c["causal"]
            frag_tokens = _estimate_tokens(frag)
            if tokens_used + frag_tokens + 2 > _MAX_TOKENS:
                break
            causal_parts.append(frag)
            tokens_used += frag_tokens + 1  # +1 for separator

        # Incorporate scenario notice if present and room allows
        if scenario_notice and scenario_notice.strip():
            sc_tokens = _estimate_tokens(scenario_notice)
            if tokens_used + sc_tokens + 2 <= _MAX_TOKENS:
                causal_parts.append(scenario_notice.strip())

        if not causal_parts:
            # No constraints — return just the intent (still better than nothing)
            return intent_phrase + "."

        return intent_phrase + ", " + ". ".join(causal_parts) + "."

    # ------------------------------------------------------------------
    # INSTRUCTION channel (FLAT mode): same info, no causal structure
    # ------------------------------------------------------------------

    def _render_instruction_flat(self, nav_command, constraints, scenario_notice=""):
        """Render FLAT instruction: same facts as causal mode but disconnected.

        Instead of "Turn left, BUT yield to pedestrian crossing 12m BEFORE turn"
        produces  "Turn left. Pedestrian 12m ahead. Pedestrian is crossing."

        Same information quantity, zero causal structure — used to isolate
        IG_structure from IG_info in the information-theoretic decomposition.
        """
        intent_phrase = _INTENT_PHRASES.get(nav_command, "Follow lane")
        tokens_used = _estimate_tokens(intent_phrase) + 1  # +1 for period

        flat_parts = [intent_phrase + "."]

        for c in constraints:
            if c["priority"] <= 0:
                continue
            # Convert causal text to flat factual statements
            frag = _flatten_constraint(c)
            frag_tokens = _estimate_tokens(frag)
            if tokens_used + frag_tokens + 1 > _MAX_TOKENS:
                break
            flat_parts.append(frag)
            tokens_used += frag_tokens + 1

        if scenario_notice and scenario_notice.strip():
            sc_tokens = _estimate_tokens(scenario_notice)
            if tokens_used + sc_tokens + 1 <= _MAX_TOKENS:
                flat_parts.append(scenario_notice.strip())

        return " ".join(flat_parts)

    # ------------------------------------------------------------------
    # NOTICE channel: Scene Graph Serialization
    # ------------------------------------------------------------------

    def _render_notice(self, state):
        """Render structured scene context for the notice channel.

        Format: [EGO] speed. [ROAD] topology. [ACTORS] nearby entities.
        Budget: ~58 tokens.
        """
        parts = []
        tokens_used = 0

        # [EGO] — speed and limit (~5-8 tokens)
        sl = state["speed_limit"]
        ego_spd = state["ego_speed_kmh"]
        if sl is not None:
            ego_str = f"[EGO] {ego_spd:.0f}/{sl:.0f}km/h."
        else:
            ego_str = f"[EGO] {ego_spd:.0f}km/h."
        parts.append(ego_str)
        tokens_used += _estimate_tokens(ego_str)

        # [ROAD] — junction, lanes, weather (~5-12 tokens)
        road = state["road"]
        road_items = []
        if road["in_junction"]:
            road_items.append("In junction")
        elif road["junction_ahead"]:
            road_items.append("Junction ahead")
        if road["lane_count"] > 1:
            road_items.append(f"{road['lane_count']}-lane")
        weather = state["weather"]
        if weather:
            road_items.extend(weather)
        if road_items:
            road_str = "[ROAD] " + ", ".join(road_items) + "."
            road_tokens = _estimate_tokens(road_str)
            if tokens_used + road_tokens <= _MAX_TOKENS:
                parts.append(road_str)
                tokens_used += road_tokens

        # [ACTORS] — nearby vehicles and pedestrians (~15-30 tokens)
        actor_items = []
        for veh in state["vehicles"][:2]:  # top 2 closest vehicles
            dist = veh["distance_m"]
            pos = veh["position"]
            spd = veh["speed_kmh"]
            if spd < 2.0:
                actor_items.append(f"Veh {pos} {dist:.0f}m stopped")
            else:
                actor_items.append(f"Veh {pos} {dist:.0f}m {spd:.0f}km/h")

        for ped in state["pedestrians"][:2]:  # top 2 closest pedestrians
            dist = ped["distance_m"]
            pos = ped["position"]
            crossing = ped["crossing_dir"]
            if "crossing" in crossing:
                actor_items.append(f"Ped {pos} {dist:.0f}m {crossing}")
            elif ped["speed_ms"] > 0.3:
                actor_items.append(f"Ped {pos} {dist:.0f}m walking")
            else:
                actor_items.append(f"Ped {pos} {dist:.0f}m")

        if actor_items:
            # Add actors one by one, respecting token budget
            included = []
            for item in actor_items:
                item_tokens = _estimate_tokens(item)
                if tokens_used + item_tokens + 3 > _MAX_TOKENS:
                    break
                included.append(item)
                tokens_used += item_tokens + 1
            if included:
                parts.append("[ACTORS] " + "; ".join(included) + ".")

        # [SIGNAL] — traffic light in notice too (concise)
        tl = state["traffic_light"]
        if tl is not None:
            sig_str = f"[SIGNAL] {tl['state']} {tl['distance_m']:.0f}m."
            sig_tokens = _estimate_tokens(sig_str)
            if tokens_used + sig_tokens <= _MAX_TOKENS:
                parts.append(sig_str)

        return " ".join(parts) if parts else ""

    # ------------------------------------------------------------------
    # Backward-compat: single-string render (old API)
    # ------------------------------------------------------------------

    def _render_single(self, nav_command, state, constraints):
        """Legacy single-string render for generate_scene_text()."""
        if not constraints:
            return ""
        intent = _NAV_COMMANDS.get(nav_command, "follow lane")
        road = state["road"]
        sl = state["speed_limit"]
        ego_spd = state["ego_speed_kmh"]

        scene_parts = []
        if sl is not None:
            scene_parts.append(f"{ego_spd:.0f}/{sl:.0f}km/h")
        else:
            scene_parts.append(f"{ego_spd:.0f}km/h")
        if road["in_junction"]:
            scene_parts.append("in junction")
        elif road["junction_ahead"]:
            scene_parts.append("junction ahead")
        if road["lane_count"] > 1:
            scene_parts.append(f"{road['lane_count']}-lane")
        weather = state["weather"]
        if weather:
            scene_parts.append(", ".join(weather))
        scene_text = "[SCENE] " + ", ".join(scene_parts) + "."

        constraint_frags = []
        tok = _estimate_tokens(scene_text)
        for c in constraints:
            if c["priority"] <= 0:
                continue
            frag = f"{c['text_short']}, {c['causal']}"
            ft = _estimate_tokens(frag)
            if tok + ft + 3 > _MAX_TOKENS:
                break
            constraint_frags.append(frag)
            tok += ft + 1
        if not constraint_frags:
            return scene_text
        return scene_text + " [CONSTRAINT] " + "; ".join(constraint_frags) + "."


# ===================================================================
# Module-level helpers
# ===================================================================

def _tl_state_str(state):
    if carla is None:
        return ""
    if state == carla.TrafficLightState.Red:
        return "Red"
    if state == carla.TrafficLightState.Yellow:
        return "Yellow"
    if state == carla.TrafficLightState.Green:
        return "Green"
    return ""


def _position_label(dot_fwd, dot_right):
    if abs(dot_right) < 3.0:
        return "ahead" if dot_fwd >= 0 else "behind"
    if dot_right > 0:
        if dot_fwd > 5:
            return "ahead-right"
        if dot_fwd < -5:
            return "behind-right"
        return "right"
    if dot_fwd > 5:
        return "ahead-left"
    if dot_fwd < -5:
        return "behind-left"
    return "left"


def _count_lanes(waypoint, direction):
    count = 0
    last_id = waypoint.lane_id
    wp = waypoint
    while True:
        wp = (
            wp.get_left_lane() if direction == "left" else wp.get_right_lane()
        )
        if wp is None:
            break
        if direction == "left" and last_id * wp.lane_id < 0:
            break
        if str(wp.lane_type) != "Driving":
            break
        last_id = wp.lane_id
        count += 1
    return count


def _vehicle_causal(intent, nav_cmd, position, is_stopped, is_approaching, dist):
    """Generate a causal connector linking a vehicle to navigation intent."""
    if nav_cmd == 1 and position in ("ahead", "ahead-left"):
        if is_approaching:
            return f"BUT oncoming vehicle {dist:.0f}m, YIELD BEFORE {intent}"
        if is_stopped:
            return f"BUT vehicle blocking {dist:.0f}m, wait BEFORE {intent}"
        return f"BUT vehicle ahead {dist:.0f}m, watch BEFORE {intent}"

    if nav_cmd == 2 and position in ("ahead", "ahead-right", "right"):
        if is_stopped:
            return f"BUT vehicle blocking {dist:.0f}m, wait BEFORE {intent}"
        return f"BUT vehicle {position} {dist:.0f}m, YIELD BEFORE {intent}"

    if nav_cmd == 5 and "left" in position:
        return f"BUT vehicle in target lane {dist:.0f}m, YIELD BEFORE {intent}"
    if nav_cmd == 6 and "right" in position:
        return f"BUT vehicle in target lane {dist:.0f}m, YIELD BEFORE {intent}"

    if position == "ahead":
        if is_stopped:
            return f"BECAUSE vehicle stopped {dist:.0f}m ahead, slow down"
        if is_approaching:
            return f"BECAUSE closing on vehicle {dist:.0f}m ahead, adjust speed"
        return f"vehicle ahead {dist:.0f}m, maintain distance"
    return f"vehicle {position} {dist:.0f}m, monitor"


def _pedestrian_causal(intent, position, is_crossing, crossing_dir, dist):
    """Generate a causal connector linking a pedestrian to navigation intent."""
    if is_crossing:
        if dist < 15:
            return f"BUT pedestrian {crossing_dir} {dist:.0f}m, YIELD BEFORE {intent}"
        return f"pedestrian {crossing_dir} {dist:.0f}m ahead, prepare to yield"
    if "ahead" in position and dist < 10:
        return f"BUT pedestrian {dist:.0f}m ahead, slow down BEFORE {intent}"
    return f"pedestrian {position} {dist:.0f}m, monitor"


def _flatten_constraint(constraint):
    """Convert a causal constraint into disconnected factual statements.

    Strips causal connectors (BUT, YIELD BEFORE, BECAUSE) and intent references,
    keeping only the factual content: object type, distance, speed, state.

    Examples:
        causal:  "BUT red light 50m ahead, stop BEFORE turn left"
        flat:    "Red light 50m ahead."

        causal:  "BUT oncoming vehicle 20m, YIELD BEFORE turn left"
        flat:    "Vehicle 20m ahead. Vehicle is approaching."

        causal:  "BUT pedestrian crossing right-to-left 12m, YIELD BEFORE turn left"
        flat:    "Pedestrian 12m ahead. Pedestrian is crossing."
    """
    kind = constraint["kind"]
    short = constraint["text_short"]

    if kind == "traffic_light":
        # "Red 50m" -> "Red light 50m ahead."
        return short.replace("Red", "Red light").replace("Yellow", "Yellow light") + " ahead."

    if kind == "vehicle":
        # "veh ahead 25m stopped" -> "Vehicle ahead 25m. Vehicle is stopped."
        parts = short.replace("veh ", "Vehicle ")
        if "stopped" in parts:
            return parts.replace(" stopped", ".") + " Vehicle is stopped."
        return parts + "."

    if kind == "pedestrian":
        # "ped ahead 12m crossing right-to-left" -> "Pedestrian ahead 12m. Pedestrian is crossing."
        parts = short.replace("ped ", "Pedestrian ")
        if "crossing" in parts:
            idx = parts.find("crossing")
            return parts[:idx].strip() + ". Pedestrian is crossing."
        return parts + "."

    if kind == "speed":
        # "30/25km/h over" -> "Current speed exceeds limit."
        return "Speed exceeds limit."

    return short + "."


def _estimate_tokens(text):
    """Rough LLaMA token count (avg ~4 chars/token for English)."""
    return max(1, (len(text) + 2) // 4)
