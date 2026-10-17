"""
DriveLM-style text enrichment for LMDrive baseline comparison.

Reproduces the text-structural pattern from DriveLM (Sima et al., 2024) by
applying its Graph Visual Question Answering (GVQA) format on top of LMDrive's
text channels. Logical dependencies between perception, prediction, and
planning are encoded with the Context: prefix that DriveLM uses to chain
parent-node QA pairs to child nodes (DriveLM §3.1).

Notice channel format (Perception P1 + Prediction P2):
    Q1 objects? A1 <object1>; <object2>; ...
    Q2 (Context: A1) future? A2 <prediction>.

Instruction channel format (Planning P3 + Behavior B):
    Q3 (Context: A1, A2) action? A3 <action sequence>.
    behavior speed=<bin> steer=<bin>

Causal information is encoded through DriveLM's graph dependencies (Context:
prefix) rather than CSN's BUT/BEFORE/BECAUSE connectives. Token budget matches
CSN.
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
# DriveLM behavior bins (cf. DriveLM §3.2 — discretized speed/steer categories)
# ---------------------------------------------------------------------------
_STEER_BIN_BY_NAV = {
    1: "left",
    2: "right",
    3: "center",
    4: "center",
    5: "left",
    6: "right",
}

_ACTION_BY_NAV = {
    1: "turn_left",
    2: "turn_right",
    3: "go_straight",
    4: "follow_lane",
    5: "lane_change_left",
    6: "lane_change_right",
}


class DriveLMStyleTextEnrichment(SceneTextEnrichment):
    """DriveLM-style enrichment built on LMDrive's text input.

    Inherits scene-state extraction and constraint scoring so the only
    experimental variable is the rendered text format.
    """

    # ------------------------------------------------------------------
    # NOTICE channel: Perception (Q1/A1) + Prediction (Q2/A2)
    # ------------------------------------------------------------------
    def _render_notice(self, state):
        # ---- Q1/A1: object enumeration ----
        a1_items = []
        for veh in state["vehicles"][:2]:
            dist = veh["distance_m"]
            pos = veh["position"]
            spd = veh["speed_kmh"]
            if spd < 2.0:
                a1_items.append(f"sedan {pos} {dist:.0f}m stopped")
            else:
                a1_items.append(f"sedan {pos} {dist:.0f}m")
        for ped in state["pedestrians"][:2]:
            dist = ped["distance_m"]
            pos = ped["position"]
            a1_items.append(f"pedestrian {pos} {dist:.0f}m")
        tl = state["traffic_light"]
        if tl is not None:
            a1_items.append(f"{tl['state'].lower()} light {tl['distance_m']:.0f}m")

        if a1_items:
            q1_block = "Q1 objects? A1 " + "; ".join(a1_items) + "."
        else:
            q1_block = "Q1 objects? A1 none."
        tokens_used = _estimate_tokens(q1_block)

        # ---- Q2/A2: prediction conditioned on A1 ----
        a2_items = self._predict_from_state(state)
        if a2_items:
            q2_text = ", ".join(a2_items)
        else:
            q2_text = "actors maintain trajectory"
        q2_block = f"Q2 (Context: A1) future? A2 {q2_text}."
        q2_tokens = _estimate_tokens(q2_block)

        if tokens_used + q2_tokens + 1 <= _MAX_TOKENS:
            return q1_block + " " + q2_block
        return q1_block

    # ------------------------------------------------------------------
    # INSTRUCTION channel: Planning (Q3/A3) + Behavior (B)
    # ------------------------------------------------------------------
    def _render_instruction(self, nav_command, constraints, scenario_notice=""):
        intent_phrase = _INTENT_PHRASES.get(nav_command, "Follow lane")
        action = _ACTION_BY_NAV.get(nav_command, "follow_lane")
        steer_bin = _STEER_BIN_BY_NAV.get(nav_command, "center")

        # Speed bin from constraints: anything blocking/yielding => slow
        slow_required = False
        wait_required = False
        for c in constraints:
            if c["priority"] <= 0:
                continue
            short = c.get("text_short", "")
            kind = c["kind"]
            if kind == "traffic_light":
                if "Red" in short or "Yellow" in short:
                    slow_required = True
                    wait_required = True
            elif kind == "pedestrian" and "crossing" in short:
                slow_required = True
                wait_required = True
            elif kind == "vehicle" and "stopped" in short:
                slow_required = True
                wait_required = True
            elif kind == "speed":
                slow_required = True
        speed_bin = "slow" if slow_required else "normal"

        # ---- Q3/A3: action with Context: A1, A2 ----
        if wait_required:
            a3_text = f"slow down, wait, then {action}"
        elif slow_required:
            a3_text = f"slow down, {action}"
        else:
            a3_text = action
        q3_block = f"Q3 (Context: A1, A2) action? A3 {a3_text}."
        behavior_block = f"behavior speed={speed_bin} steer={steer_bin}."

        tokens_used = _estimate_tokens(q3_block) + _estimate_tokens(behavior_block) + 1
        if tokens_used <= _MAX_TOKENS:
            return q3_block + " " + behavior_block
        return q3_block

    # FLAT mode reuses the structured renderer; the comparison axis is
    # structure type (graph vs. CoT vs. connectives).
    def _render_instruction_flat(self, nav_command, constraints, scenario_notice=""):
        return self._render_instruction(nav_command, constraints, scenario_notice)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _predict_from_state(self, state):
        """Generate short prediction phrases for the A2 answer."""
        preds = []
        for ped in state["pedestrians"][:2]:
            cd = ped.get("crossing_dir", "")
            if "crossing" in cd:
                preds.append("ped may cross route")
                break
        for veh in state["vehicles"][:2]:
            if veh["speed_kmh"] < 2.0:
                preds.append("sedan remains stopped")
                break
            if veh["rel_speed_kmh"] > 5.0:
                preds.append("ego closing on sedan")
                break
        tl = state["traffic_light"]
        if tl is not None and tl["state"] in ("Red", "Yellow"):
            preds.append("light requires stop")
        return preds
