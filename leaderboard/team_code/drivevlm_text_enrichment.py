"""
DriveVLM-style text enrichment for LMDrive baseline comparison.

Reproduces the text-structural pattern from DriveVLM (Tian et al., 2024) by
applying its three-stage Chain-of-Thought format (Scene Description, Scene
Analysis, Hierarchical Planning) on top of LMDrive's text channels. The
underlying VLA, scene state, and constraint identification are identical to
CSN; only the text rendering differs.

Notice channel format (mirrors DriveVLM's Scene Description, paper Figure 1):
    [ENV] <weather/time/road_type>
    [OBJ] <object1>; <object2>; ...

Instruction channel format (mirrors DriveVLM's Scene Analysis + Hierarchical
Planning, paper sections 3.3 and 3.4):
    [ANALYSIS]
    <obj>: motion=<state>; influence=<label>
    ...
    [PLAN]
    meta=[<actions>]
    decision=<short text>

Causal information is encoded through DriveVLM's Influence: field rather than
through CSN's BUT/BEFORE/BECAUSE connectives.

============================================================================
Faithfulness note
============================================================================
DriveVLM section 3.3 defines Critical Object Analysis with three attribute
classes per object: static attributes (C_s), motion states (C_m), and
particular behaviors (C_b), followed by an influence prediction I. Paper
Figure 1 renders this as three lines per object:

    Critical Object: Police car
    Characteristic:  Parking on the right side
    Influence:       Blocking the right lane

We adopt a single-line compressed form `obj: motion=<C_m>; influence=<I>` for
two reasons. First, LMDrive's notice/instruction channels are hard-capped at
64 tokens each, so the verbatim three-line format would fit only one or two
objects per frame, defeating the comparison. Second, paper section 3.3
explicitly notes that 'only one or two characteristics apply to a critical
object' in practice, and we keep the most informative one (C_m) while still
preserving the influence prediction step that characterises DriveVLM. All
other baselines (CSN, DriveLM-style) operate under the same token budget,
giving an apples-to-apples comparison of structural patterns rather than of
verbosity.

The verbatim three-line format, with fewer objects per frame, is implemented
in drivevlm_faithful_text_enrichment.py.
============================================================================
"""

import logging

from team_code.scene_text_enrichment import (
    SceneTextEnrichment,
    _NAV_COMMANDS,
    _INTENT_PHRASES,
    _CONFLICT_SIDES,
    _estimate_tokens,
    _MAX_TOKENS,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Meta-action tokens used by DriveVLM
# ---------------------------------------------------------------------------
_META_ACTIONS_BY_NAV = {
    1: "turn_left",
    2: "turn_right",
    3: "go_straight",
    4: "follow_lane",
    5: "lane_change_left",
    6: "lane_change_right",
}


class DriveVLMStyleTextEnrichment(SceneTextEnrichment):
    """DriveVLM-style enrichment built on LMDrive's text input.

    Inherits scene-state extraction and constraint scoring from
    SceneTextEnrichment so that the only experimental variable is the rendered
    text format.
    """

    # ------------------------------------------------------------------
    # NOTICE channel: Scene Description (DriveVLM Stage 1)
    # ------------------------------------------------------------------
    def _render_notice(self, state):
        parts = []
        tokens_used = 0

        # [ENV] block: weather/time/road_type
        env_items = []
        weather = state["weather"]
        if weather:
            env_items.extend(weather)
        else:
            env_items.append("clear")
        road = state["road"]
        if road["in_junction"]:
            env_items.append("junction")
        elif road["junction_ahead"]:
            env_items.append("junction-ahead")
        elif road["lane_count"] > 1:
            env_items.append(f"{road['lane_count']}-lane")
        else:
            env_items.append("urban")
        env_str = "[ENV] " + " ".join(env_items) + "."
        parts.append(env_str)
        tokens_used += _estimate_tokens(env_str)

        # [OBJ] block: critical objects with bounding-box-equivalent quantitative facts
        obj_items = []
        for veh in state["vehicles"][:2]:
            dist = veh["distance_m"]
            pos = veh["position"]
            spd = veh["speed_kmh"]
            if spd < 2.0:
                obj_items.append(f"sedan {pos} {dist:.0f}m stopped")
            else:
                obj_items.append(f"sedan {pos} {dist:.0f}m {spd:.0f}km/h")
        for ped in state["pedestrians"][:2]:
            dist = ped["distance_m"]
            pos = ped["position"]
            spd = ped["speed_ms"]
            if spd > 0.3:
                obj_items.append(f"ped {pos} {dist:.0f}m {spd:.1f}m/s")
            else:
                obj_items.append(f"ped {pos} {dist:.0f}m static")

        # Traffic light treated as a critical object
        tl = state["traffic_light"]
        if tl is not None:
            obj_items.append(f"light {tl['distance_m']:.0f}m {tl['state'].lower()}")

        if obj_items:
            included = []
            for it in obj_items:
                t = _estimate_tokens(it)
                if tokens_used + t + 3 > _MAX_TOKENS:
                    break
                included.append(it)
                tokens_used += t + 1
            if included:
                parts.append("[OBJ] " + "; ".join(included) + ".")

        return " ".join(parts) if parts else ""

    # ------------------------------------------------------------------
    # INSTRUCTION channel: Scene Analysis + Hierarchical Planning
    # (DriveVLM Stages 2 + 3)
    # ------------------------------------------------------------------
    def _render_instruction(self, nav_command, constraints, scenario_notice=""):
        intent_phrase = _INTENT_PHRASES.get(nav_command, "Follow lane")
        meta_action = _META_ACTIONS_BY_NAV.get(nav_command, "follow_lane")
        conflict_sides = _CONFLICT_SIDES.get(nav_command, {"ahead"})

        # ANALYSIS lines: motion + influence per critical constraint
        analysis_lines = []
        slow_required = False
        wait_required = False
        for c in constraints:
            if c["priority"] <= 0:
                continue
            line = self._analysis_line(c, conflict_sides)
            if line is None:
                continue
            analysis_lines.append(line)
            # Pick up meta-action hints from blocking/temporal constraints
            short = c.get("text_short", "")
            if "stopped" in short or "crossing" in short:
                slow_required = True
                wait_required = True
            elif "BUT" in c.get("causal", ""):
                slow_required = True

        # Build PLAN block
        meta_actions = []
        if slow_required:
            meta_actions.append("slow_down")
        meta_actions.append(meta_action)
        plan_block = (
            f"[PLAN] meta=[{','.join(meta_actions)}] "
            f"decision={intent_phrase}"
            f"{' after yielding' if wait_required else ''}."
        )

        # Assemble within token budget
        body = "[ANALYSIS] "
        tokens_used = _estimate_tokens(body) + _estimate_tokens(plan_block) + 2
        chosen = []
        for line in analysis_lines:
            t = _estimate_tokens(line)
            if tokens_used + t + 1 > _MAX_TOKENS:
                break
            chosen.append(line)
            tokens_used += t + 1

        if chosen:
            return body + " ".join(chosen) + " " + plan_block
        # No analysis fits — fall back to plan-only output
        return plan_block

    # FLAT mode reuses the structured renderer; the comparison axis we care
    # about is structure type, not flat-vs-structured.
    def _render_instruction_flat(self, nav_command, constraints, scenario_notice=""):
        return self._render_instruction(nav_command, constraints, scenario_notice)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _analysis_line(self, constraint, conflict_sides):
        """Render one ANALYSIS line in DriveVLM's motion=<x>; influence=<y> format.

        Any constraint reaching this function has positive priority, so it is
        relevant to the ego maneuver. The Influence label distinguishes the
        kind of conflict (must-stop, must-yield, blocks-path, affects-ego).
        """
        kind = constraint["kind"]
        short = constraint.get("text_short", "")

        if kind == "vehicle":
            tokens = short.split()
            position = tokens[1] if len(tokens) > 1 else "ahead"
            is_stopped = "stopped" in short
            motion = "stopped" if is_stopped else "moving"
            on_conflict = position in conflict_sides
            if is_stopped and (on_conflict or "ahead" in position):
                influence = "blocks-path"
            elif on_conflict:
                influence = "must-yield"
            else:
                influence = "affects-ego"
            return f"sedan: motion={motion}; influence={influence}."

        if kind == "pedestrian":
            tokens = short.split()
            position = tokens[1] if len(tokens) > 1 else "ahead"
            is_crossing = "crossing" in short
            motion = "crossing" if is_crossing else "static"
            if is_crossing:
                influence = "must-yield"
            elif position in conflict_sides:
                influence = "affects-ego"
            else:
                influence = "affects-ego"
            return f"ped: motion={motion}; influence={influence}."

        if kind == "traffic_light":
            state = short.split()[0].lower() if short else "unknown"
            influence = "must-stop" if state in ("red", "yellow") else "affects-ego"
            return f"light: state={state}; influence={influence}."

        if kind == "speed":
            return "speed: state=over-limit; influence=must-slow."

        return None
