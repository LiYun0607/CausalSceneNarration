"""Agent configuration used for every closed-loop run in the paper.

Pass this file to the leaderboard evaluator with --agent-config. Paths are read
from environment variables so the same file serves every configuration:

    LMDRIVE_LLM_MODEL      directory of liuhaotian/llava-v1.5-7b
    LMDRIVE_VISION_CKPT    vision-encoder-r50.pth.tar
                           (OpenDILabCommunity/LMDrive-vision-encoder-r50-v1.0)
    LMDRIVE_LMDRIVE_CKPT   llava-v1.5-checkpoint.pth
                           (OpenDILabCommunity/LMDrive-llava-v1.5-7b-v1.0)
    LMDRIVE_LORA_PATH      directory of the preference-tuned LoRA adapter
                           ("PL-DPO-NLL" rows), or "none" for the original
                           LMDrive weights. Default: none.

The controller and notice settings are the ones used in the paper.
"""

import os


def _env_path(name, default):
    value = os.environ.get(name, default)
    return os.path.expanduser(value) if value else value


class GlobalConfig:
    def __init__(self):
        # Model files
        self.llm_model = _env_path("LMDRIVE_LLM_MODEL", "checkpoints/llava-v1.5-7b")
        self.preception_model = "memfuser_baseline_e1d3_return_feature"
        self.preception_model_ckpt = _env_path(
            "LMDRIVE_VISION_CKPT", "checkpoints/vision-encoder-r50.pth.tar")
        self.lmdrive_ckpt = _env_path(
            "LMDRIVE_LMDRIVE_CKPT", "checkpoints/llava-v1.5-checkpoint.pth")

        # LoRA adapter (None = original LMDrive weights)
        lora = os.environ.get("LMDRIVE_LORA_PATH", "none").strip()
        self.lora_path = None if lora.lower() in ("", "none") else os.path.expanduser(lora)

        # PID controller (LMDrive defaults)
        self.turn_KP = 1.25
        self.turn_KI = 0.75
        self.turn_KD = 0.3
        self.turn_n = 40

        self.speed_KP = 5.0
        self.speed_KI = 0.5
        self.speed_KD = 1.0
        self.speed_n = 40

        self.max_throttle = 0.75
        self.brake_speed = 0.1
        self.brake_ratio = 1.1
        self.clip_delta = 0.35

        # Language inputs: the notice channel is used (CSN text enters here)
        self.agent_use_notice = True
        self.sample_rate = 2
