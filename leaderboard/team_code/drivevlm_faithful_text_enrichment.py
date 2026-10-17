"""
DriveVLM-faithful text enrichment — verbatim three-line Critical Object
Analysis from DriveVLM (Tian et al., 2024) Figure 1, applied to LMDrive.

This is the verbatim companion to drivevlm_text_enrichment.py (compressed
single-line form). It tests whether the compressed form lost DriveVLM's
static (C_s) and behavior (C_b) attributes. This module reproduces the
DriveVLM paper's Figure 1 layout almost verbatim, expanding all three attribute classes
(static / motion / particular behavior) and using natural-language Influence
phrases instead of enum labels.

Notice channel — Stage 1: Scene Description (paper section 3.2):
    Weather: <env>
    Road type: <topology>
    Time: <day/night>
    Lane condition: <lanes>
    Critical objects: <type1> at (x1,y1), <type2> at (x2,y2)

Instruction channel — Stage 2 + Stage 3 (paper sections 3.3 + 3.4):
    Critical Object: <type>
    Characteristic:  <C_s | C_m | C_b natural-language phrase>
    Influence:       <natural-language phrase, e.g. 'Blocking the lane'>
    ...
    Meta-actions: [<action1>, <action2>]
    Decision:     <action><subject><duration>
    Waypoints:    (skipped — LMDrive predicts these directly)

Token budget is relaxed: notice 64-tok hard cap is preserved, instruction
allowed to use the full 64 tokens for analysis + plan, since DriveVLM's
natural-language form is intrinsically more verbose. The compressed sibling
(drivevlm_text_enrichment.py) and this faithful version bracket the
performance question.
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


# Paper section 3.4 lists 17 meta-action categories. We use a representative
# subset matching the navigation command vocabulary.
_META_ACTION_BY_NAV = {
    1: "turn left",
    2: "turn right",
    3: "go straight",
    4: "follow lane",
    5: "lane change left",
    6: "lane change right",
}

# DriveVLM allows 64-token notice; instruction is allowed up to 64 tokens too
# because the paper's natural-language form needs the room. Both still fit
# LMDrive's hard input limit.
_NOTICE_BUDGET = 60
_INSTR_BUDGET = 60


class DriveVLMFaithfulTextEnrichment(SceneTextEnrichment):
    """Verbatim DriveVLM three-line Critical Object Analysis style."""

    # ------------------------------------------------------------------
    # NOTICE channel: Scene Description with E_weather / E_road / E_time / E_lane
    # ------------------------------------------------------------------
    def _render_notice(self, state):
        weather = state["weather"]
        if "night" in weather:
            time_str = "Time: night."
            weather_str = "Weather: " + ", ".join(
                w for w in weather if w != "night"
            ) + "." if any(w != "night" for w in weather) else "Weather: clear."
        else:
            time_str = "Time: day."
            weather_str = "Weather: " + (", ".join(weather) if weather else "clear") + "."

        road = state["road"]
        if road["in_junction"]:
            road_type = "Road type: junction."
        elif road["junction_ahead"]:
            road_type = "Road type: approaching junction."
        elif road["lane_count"] > 1:
            road_type = f"Road type: urban {road['lane_count']}-lane."
        else:
            road_type = "Road type: urban single lane."

        if road["lane_count"] >= 2:
            lane_cond = f"Lane condition: {road['lane_count']} lanes available."
        else:
            lane_cond = "Lane condition: single lane."

        # Critical-object enumeration with approximate egocentric coordinates
        crit_items = []
        for veh in state["vehicles"][:2]:
            dx = veh.get("dot_fwd", 0)
            dy = veh.get("dot_right", 0)
            crit_items.append(f"sedan at ({dx:.0f},{dy:.0f})")
        for ped in state["pedestrians"][:2]:
            dx = ped.get("dot_fwd", 0)
            dy = ped.get("dot_right", 0)
            crit_items.append(f"pedestrian at ({dx:.0f},{dy:.0f})")
        tl = state["traffic_light"]
        if tl is not None:
            crit_items.append(f"{tl['state'].lower()} light at {tl['distance_m']:.0f}m")
        crit_str = "Critical objects: " + (
            "; ".join(crit_items) if crit_items else "none"
        ) + "."

        # Assemble within budget
        parts = [weather_str, road_type, time_str, lane_cond, crit_str]
        out, used = [], 0
        for p in parts:
            t = _estimate_tokens(p)
            if used + t + 1 > _NOTICE_BUDGET:
                break
            out.append(p)
            used += t + 1
        return " ".join(out)

    # ------------------------------------------------------------------
    # INSTRUCTION channel: 3-line Critical Object Analysis + Hierarchical Plan
    # ------------------------------------------------------------------
    def _render_instruction(self, nav_command, constraints, scenario_notice=""):
        meta_action = _META_ACTION_BY_NAV.get(nav_command, "follow lane")
        conflict_sides = _CONFLICT_SIDES.get(nav_command, {"ahead"})

        analysis_blocks = []
        slow_required = False
        wait_required = False
        wait_subject = "intersection"

        for c in constraints:
            if c["priority"] <= 0:
                continue
            block = self._three_line_block(c, conflict_sides)
            if block is None:
                continue
            analysis_blocks.append(block)
            short = c.get("text_short", "")
            kind = c["kind"]
            if kind == "traffic_light" and ("Red" in short or "Yellow" in short):
                slow_required = True
                wait_required = True
                wait_subject = "the light"
            elif kind == "pedestrian" and "crossing" in short:
                slow_required = True
                wait_required = True
                wait_subject = "the pedestrian"
            elif kind == "vehicle" and "stopped" in short:
                slow_required = True
                wait_required = True
                wait_subject = "the vehicle"
            elif kind == "speed":
                slow_required = True

        # Hierarchical Planning block (paper section 3.4):
        #   Meta-actions A, Decision D = (Action, Subject, Duration)
        meta_actions = []
        if slow_required:
            meta_actions.append("decelerate")
        meta_actions.append(meta_action)
        meta_str = "Meta: [" + ", ".join(meta_actions) + "]."

        if wait_required:
            decision = f"Decision: wait for {wait_subject}, then {meta_action}."
        elif slow_required:
            decision = f"Decision: slow, then {meta_action}."
        else:
            decision = f"Decision: {meta_action}."

        # Token budgeting: keep the meta+decision tail; drop low-priority
        # analysis blocks if we run out of room.
        tail = meta_str + " " + decision
        tail_tok = _estimate_tokens(tail) + 1
        chosen = []
        used = tail_tok
        for blk in analysis_blocks:
            t = _estimate_tokens(blk)
            if used + t + 1 > _INSTR_BUDGET:
                break
            chosen.append(blk)
            used += t + 1

        if chosen:
            return " ".join(chosen) + " " + tail
        return tail

    def _render_instruction_flat(self, nav_command, constraints, scenario_notice=""):
        return self._render_instruction(nav_command, constraints, scenario_notice)

    # ------------------------------------------------------------------
    # Helpers — three-line block per critical object
    # ------------------------------------------------------------------
    def _three_line_block(self, constraint, conflict_sides):
        """Render `Critical Object / Characteristic / Influence` for one entity.

        Mirrors paper Figure 1 verbatim. Characteristic prose chooses among
        C_s (static), C_m (motion), C_b (particular behavior) by entity type
        and dynamic state, following paper section 3.3 ('only one or two
        characteristics apply to a critical object'). Prose is kept terse so
        2-3 blocks fit alongside the planning block within the token budget.
        """
        kind = constraint["kind"]
        short = constraint.get("text_short", "")

        if kind == "vehicle":
            tokens = short.split()
            position = tokens[1] if len(tokens) > 1 else "ahead"
            dist = self._extract_distance(short, default=20)
            is_stopped = "stopped" in short
            on_conflict = position in conflict_sides
            obj = "Object: sedan."
            if is_stopped:
                char = f"Characteristic: stopped in lane {position}, {dist:.0f}m."
                inf = ("Influence: blocks ego path." if on_conflict
                       else "Influence: adjacent lane, monitor.")
            else:
                speed_kmh = self._extract_speed(short, default=30)
                char = f"Characteristic: moving {speed_kmh:.0f} km/h, {position} {dist:.0f}m."
                inf = ("Influence: ego must yield until clear." if on_conflict
                       else "Influence: no direct conflict.")
            return f"{obj} {char} {inf}"

        if kind == "pedestrian":
            tokens = short.split()
            position = tokens[1] if len(tokens) > 1 else "ahead"
            dist = self._extract_distance(short, default=10)
            is_crossing = "crossing" in short
            obj = "Object: pedestrian."
            if is_crossing:
                if "right-to-left" in short:
                    cdir = "right to left"
                elif "left-to-right" in short:
                    cdir = "left to right"
                else:
                    cdir = "across road"
                char = f"Characteristic: crossing {cdir}, {position} {dist:.0f}m."
                inf = "Influence: ego must yield before turning."
            else:
                char = f"Characteristic: standing on sidewalk, {position} {dist:.0f}m."
                inf = "Influence: monitor for sudden movement."
            return f"{obj} {char} {inf}"

        if kind == "traffic_light":
            tokens = short.split()
            state = tokens[0].lower() if tokens else "unknown"
            dist = self._extract_distance(short, default=20)
            obj = "Object: traffic light."
            char = f"Characteristic: {state} signal, {dist:.0f}m ahead."
            if state == "red":
                inf = "Influence: ego must stop at line."
            elif state == "yellow":
                inf = "Influence: ego prepares to stop."
            else:
                inf = "Influence: ego may proceed."
            return f"{obj} {char} {inf}"

        if kind == "speed":
            obj = "Object: speed limit."
            char = "Characteristic: ego exceeds posted limit."
            inf = "Influence: ego must decelerate."
            return f"{obj} {char} {inf}"

        return None

    @staticmethod
    def _extract_distance(text, default=10):
        import re
        m = re.search(r'(\d+\.?\d*)m', text)
        return float(m.group(1)) if m else default

    @staticmethod
    def _extract_speed(text, default=30):
        import re
        m = re.search(r'(\d+\.?\d*)km/h', text)
        return float(m.group(1)) if m else default
