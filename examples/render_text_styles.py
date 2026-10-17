"""
Render every text style of the paper for one synthetic left-turn scene.

Runs without CARLA or a GPU: the CARLA-dependent scene extractor is replaced by
a fixed scene state (pedestrian crossing from the right at 5 m, sedan ahead at
12 m, red light at 20 m, 25/30 km/h, junction ahead).

Run from the repository root:
    python examples/render_text_styles.py

In the closed-loop agent only the first string is sent to LMDrive, through its
notice channel, and only while LMDrive's own scenario notice is inactive; the
second string is used only when the first one is empty. The
token counts are the renderer's estimate (len(text) + 2) // 4, which is lower
than the LLaMA token count; LMDrive keeps the last 64 LLaMA tokens of the notice.
See README, "Notes on the code".
"""

import sys
from pathlib import Path

# Make team_code importable when running this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "leaderboard"))


# ---------------------------------------------------------------------------
# Synthetic scene state (left-turn example)
# ---------------------------------------------------------------------------
SYNTH_STATE = {
    "ego_speed_kmh": 25.0,
    "ego_location": None,
    "ego_fwd": None,
    "ego_right": None,
    "traffic_light": {"state": "Red", "distance_m": 20.0},
    "speed_limit": 30.0,
    "vehicles": [
        {
            "distance_m": 12.0,
            "position": "ahead",
            "speed_kmh": 30.0,
            "rel_speed_kmh": -5.0,
            "dot_fwd": 12.0,
            "dot_right": 0.0,
        }
    ],
    "pedestrians": [
        {
            "distance_m": 5.0,
            "position": "right",
            "speed_ms": 1.5,
            "crossing_dir": "crossing right-to-left",
            "dot_fwd": 0.0,
            "dot_right": 5.0,
        }
    ],
    "road": {"in_junction": False, "junction_ahead": True, "lane_count": 2},
    "weather": [],
}


def _patch_extractor(enricher):
    """Replace the CARLA-dependent extractor with our synthetic state."""
    enricher._extract_scene_state = lambda: SYNTH_STATE
    return enricher


def main():
    from team_code.scene_text_enrichment import SceneTextEnrichment
    from team_code.drivevlm_text_enrichment import DriveVLMStyleTextEnrichment
    from team_code.drivelm_text_enrichment import DriveLMStyleTextEnrichment
    from team_code.drivevlm_faithful_text_enrichment import DriveVLMFaithfulTextEnrichment
    from team_code.drivelm_faithful_text_enrichment import DriveLMFaithfulTextEnrichment

    enrichers = {
        "CSN": SceneTextEnrichment.__new__(SceneTextEnrichment),
        "Flat Text": SceneTextEnrichment.__new__(SceneTextEnrichment),
        "DriveVLM-style": DriveVLMStyleTextEnrichment.__new__(
            DriveVLMStyleTextEnrichment
        ),
        "DriveVLM-faithful": DriveVLMFaithfulTextEnrichment.__new__(
            DriveVLMFaithfulTextEnrichment
        ),
        "DriveLM-style": DriveLMStyleTextEnrichment.__new__(
            DriveLMStyleTextEnrichment
        ),
        "DriveLM-faithful": DriveLMFaithfulTextEnrichment.__new__(
            DriveLMFaithfulTextEnrichment
        ),
    }
    # Bypass __init__'s CARLA dependency by setting the few fields we need.
    for name, e in enrichers.items():
        e._vehicle = None
        e._world = None
        e._map = None
        e._vehicle_radius = 50.0
        e._ped_radius = 30.0
        e._max_vehicles = 3
        e._max_pedestrians = 2
        e._flat_mode = (name == "Flat Text")
        e._noise_dist_std = 0.0
        e._noise_speed_rel = 0.0
        e._noise_miss_rate = 0.0
        e._noise_rng = None
        _patch_extractor(e)

    nav_command = 1  # turn left

    for name, e in enrichers.items():
        instruction, notice = e.generate(
            nav_command=nav_command,
            base_instruction="Turn left at intersection.",
            # Empty, because a non-empty LMDrive scenario notice is sent
            # instead of the rendered text.
            scenario_notice="",
            tick_data={"next_command": 1},
        )
        print(f"=== {name} ===")
        print(f"  first string, sent as notice ({_token_count(instruction)} tok est.): {instruction}")
        print(f"  second string, fallback      ({_token_count(notice)} tok est.): {notice}")
        print()


def _token_count(text):
    return max(1, (len(text) + 2) // 4) if text else 0


if __name__ == "__main__":
    main()
