"""
DriveLM-faithful text enrichment — verbatim Graph Visual Question Answering
chain from DriveLM (Sima et al., 2024) sections 3.1-3.2, applied to LMDrive.

This is the verbatim companion to drivelm_text_enrichment.py (compressed
two-question form). It tests whether cutting the QA chain explains the
compressed result. This module expands the perception (P1) → prediction (P2) → planning (P3) chain to the
full breadth used in the paper, including object identification, motion
analysis, future prediction, safe-action planning, and behavior/motion
output (paper Table 1 lists 91+ QA per frame in DriveLM-CARLA).

Paper structure (section 3.1):
    Perception P1: identification, description, localisation of key objects
    Prediction  P2: estimation of action/interaction of key objects
    Planning    P3: possible safe actions of the ego vehicle
    Behavior    B : classification of driving decision (speed/steer bins)
    Motion      M : waypoints (handled by LMDrive directly)

DriveLM appends parent-node QA via "Context: A_parent" prefix (paper section
3.1). We reproduce this verbatim across multiple QA pairs.

Notice channel (P1 + P2):
    Q1 What objects worth noting? A1 ...
    Q2 (Context: A1) What is the motion state of <obj>? A2 ...
    Q3 (Context: A1, A2) What might <obj> do next? A3 ...

Instruction channel (P3 + B):
    Q4 (Context: A1, A2, A3) What are the safe actions of ego? A4 ...
    Q5 (Context: A4) What is the behavior? A5 speed=<bin>, steer=<bin>.
"""

import logging

from team_code.scene_text_enrichment import (
    SceneTextEnrichment,
    _NAV_COMMANDS,
    _INTENT_PHRASES,
    _CONFLICT_SIDES,
    _estimate_tokens,
)

logger = logging.getLogger(__name__)


# DriveLM section 3.2 defines speed and steer bins. We use the bin set most
# commonly cited in the paper (5 speed bins, 3 steer bins).
_STEER_BIN_BY_NAV = {
    1: "left",
    2: "right",
    3: "center",
    4: "center",
    5: "left",
    6: "right",
}


_NOTICE_BUDGET = 60
_INSTR_BUDGET = 60


class DriveLMFaithfulTextEnrichment(SceneTextEnrichment):
    """Verbatim DriveLM Graph VQA style across both channels."""

    # ------------------------------------------------------------------
    # NOTICE channel: P1 (perception) + P2 (prediction)
    # ------------------------------------------------------------------
    def _render_notice(self, state):
        # ---- Q1/A1: object identification ----
        objs = self._enumerate_objects(state)
        a1 = "; ".join(objs) if objs else "no critical objects"
        q1 = f"Q1 What objects worth noting? A1 {a1}."

        # ---- Q2/A2: motion state per key object (Context: A1) ----
        motion_phrases = self._motion_phrases(state)
        if motion_phrases:
            a2 = "; ".join(motion_phrases)
        else:
            a2 = "actors are stationary or moving at constant speed"
        q2 = f"Q2 (Context: A1) Motion state? A2 {a2}."

        # ---- Q3/A3: future prediction (Context: A1, A2) ----
        pred_phrases = self._prediction_phrases(state)
        if pred_phrases:
            a3 = "; ".join(pred_phrases)
        else:
            a3 = "actors maintain current trajectories"
        q3 = f"Q3 (Context: A1, A2) Future? A3 {a3}."

        out, used = [], 0
        for q in [q1, q2, q3]:
            t = _estimate_tokens(q)
            if used + t + 1 > _NOTICE_BUDGET:
                break
            out.append(q)
            used += t + 1
        return " ".join(out)

    # ------------------------------------------------------------------
    # INSTRUCTION channel: P3 (planning) + B (behavior)
    # ------------------------------------------------------------------
    def _render_instruction(self, nav_command, constraints, scenario_notice=""):
        intent = _NAV_COMMANDS.get(nav_command, "follow lane")
        steer_bin = _STEER_BIN_BY_NAV.get(nav_command, "center")

        slow_required = False
        wait_required = False
        for c in constraints:
            if c["priority"] <= 0:
                continue
            short = c.get("text_short", "")
            kind = c["kind"]
            if kind == "traffic_light" and ("Red" in short or "Yellow" in short):
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
        speed_bin = "stop" if wait_required else ("slow" if slow_required else "normal")

        # ---- Q4/A4: safe actions (Context: A1, A2, A3) ----
        if wait_required:
            a4 = (
                f"the ego car should slow down, wait for the conflict to clear, "
                f"then {intent}."
            )
        elif slow_required:
            a4 = f"the ego car should slow down and {intent} cautiously."
        else:
            a4 = f"the ego car should {intent} while maintaining a safe distance."
        q4 = f"Q4 (Context: A1, A2, A3) Safe actions? A4 {a4}"

        # ---- Q5/A5: behavior bins (Context: A4) ----
        q5 = f"Q5 (Context: A4) Behavior? A5 speed={speed_bin}, steer={steer_bin}."

        out, used = [], 0
        for q in [q4, q5]:
            t = _estimate_tokens(q)
            if used + t + 1 > _INSTR_BUDGET:
                break
            out.append(q)
            used += t + 1
        return " ".join(out)

    def _render_instruction_flat(self, nav_command, constraints, scenario_notice=""):
        return self._render_instruction(nav_command, constraints, scenario_notice)

    # ------------------------------------------------------------------
    # Helpers — answer-side prose for each P-stage
    # ------------------------------------------------------------------
    def _enumerate_objects(self, state):
        items = []
        for veh in state["vehicles"][:2]:
            items.append(f"a sedan {veh['position']} of ego at {veh['distance_m']:.0f}m")
        for ped in state["pedestrians"][:2]:
            items.append(f"a pedestrian {ped['position']} of ego at {ped['distance_m']:.0f}m")
        tl = state["traffic_light"]
        if tl is not None:
            items.append(f"a {tl['state'].lower()} traffic light {tl['distance_m']:.0f}m ahead")
        return items

    def _motion_phrases(self, state):
        items = []
        for veh in state["vehicles"][:2]:
            if veh["speed_kmh"] < 2.0:
                items.append(f"the sedan is stopped")
            else:
                items.append(f"the sedan is moving at {veh['speed_kmh']:.0f} km/h")
        for ped in state["pedestrians"][:2]:
            cd = ped.get("crossing_dir", "")
            if "crossing" in cd:
                items.append(f"the pedestrian is {cd.replace('-', ' ')}")
            elif ped["speed_ms"] > 0.3:
                items.append("the pedestrian is walking along the road")
            else:
                items.append("the pedestrian is stationary")
        return items

    def _prediction_phrases(self, state):
        items = []
        for ped in state["pedestrians"][:2]:
            if "crossing" in ped.get("crossing_dir", ""):
                items.append("the pedestrian is likely to cross the ego lane")
                break
        for veh in state["vehicles"][:2]:
            if veh["speed_kmh"] < 2.0:
                items.append("the sedan will remain stationary")
                break
            if veh["rel_speed_kmh"] > 5.0:
                items.append("ego is closing on the sedan ahead")
                break
        tl = state["traffic_light"]
        if tl is not None and tl["state"] in ("Red", "Yellow"):
            items.append("the light requires the ego to stop")
        return items
