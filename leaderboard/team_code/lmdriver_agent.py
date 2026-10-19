# This file is derived from leaderboard/team_code/lmdriver_agent.py of LMDrive
# (https://github.com/opendilab/LMDrive, Apache License 2.0).
# Modified for "Causal Scene Narration: Inference-Time Text Restructuring for
# Vision-Language-Action Driving" (TMLR): optional text restructuring (CSN,
# Flat Text, DriveVLM-style and DriveLM-style renderers), optional runtime
# monitors, optional LoRA adapter loading, headless display, token logging.
import os
import json
import random
import datetime
import pathlib
import time
import imp
import logging
from collections import deque
import math

import yaml
import cv2
import torch
import carla
import numpy as np
from PIL import Image
from easydict import EasyDict
from torchvision import transforms

from leaderboard.autoagents import autonomous_agent
from team_code.planner import RoutePlanner, InstructionPlanner
from team_code.pid_controller import PIDController
from timm.models import create_model
from lavis.common.registry import registry

try:
    import pygame
except ImportError:
    raise RuntimeError("cannot import pygame, make sure pygame package is installed")

# Runtime safety modules (optional, controlled by env vars)
SAFETY_ENABLED = os.environ.get("LMDRIVE_SAFETY_ENABLED", "false").lower() in ("true", "1", "yes")
TEXT_ENRICHMENT_ENABLED = os.environ.get("LMDRIVE_TEXT_ENRICHMENT", "false").lower() in ("true", "1", "yes")
TEXT_STYLE = os.environ.get("LMDRIVE_TEXT_STYLE", "csn").lower()  # csn | drivevlm | drivevlm_faithful | drivelm | drivelm_faithful

logger = logging.getLogger("LMDriveAgent")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


SAVE_PATH = os.environ.get("SAVE_PATH", 'eval')
if SAVE_PATH.strip().lower() in ("", "none"):
    SAVE_PATH = None  # do not write per-frame images
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

def rotate_lidar(lidar, angle):
    radian = np.deg2rad(angle)
    return lidar @ [
        [ np.cos(radian), np.sin(radian), 0, 0],
        [-np.sin(radian), np.cos(radian), 0, 0],
        [0,0,1,0],
        [0,0,0,1]
    ]

def lidar_to_raw_features(lidar):
    def preprocess(lidar_xyzr, lidar_painted=None):

        idx = (lidar_xyzr[:,0] > -1.2)&(lidar_xyzr[:,0] < 1.2)&(lidar_xyzr[:,1]>-1.2)&(lidar_xyzr[:,1]<1.2)

        idx = np.argwhere(idx)

        if lidar_painted is None:
            return np.delete(lidar_xyzr, idx, axis=0)
        else:
            return np.delete(lidar_xyzr, idx, axis=0), np.delete(lidar_painted, idx, axis=0)

    lidar_xyzr = preprocess(lidar)

    idxs = np.arange(len(lidar_xyzr))
    np.random.shuffle(idxs)
    lidar_xyzr = lidar_xyzr[idxs]

    lidar = np.zeros((40000, 4), dtype=np.float32)
    num_points = min(40000, len(lidar_xyzr))
    lidar[:num_points,:4] = lidar_xyzr
    lidar[np.isinf(lidar)] = 0
    lidar[np.isnan(lidar)] = 0
    lidar = rotate_lidar(lidar, -90).astype(np.float32)
    return lidar, num_points

class DisplayInterface(object):
    def __init__(self):
        self._width = 1200
        self._height = 900
        self._surface = None
        self._display = None
        self._enabled = not os.environ.get("LMDRIVE_HEADLESS", "false").lower() in ("true", "1", "yes")

        if self._enabled:
            pygame.init()
            pygame.font.init()
            self._clock = pygame.time.Clock()
            self._display = pygame.display.set_mode(
                (self._width, self._height), pygame.HWSURFACE | pygame.DOUBLEBUF
            )
            pygame.display.set_caption("LMDrive Agent")
        else:
            print("[DisplayInterface] Headless mode, skipping pygame display")

    def run_interface(self, input_data):
        surface = np.zeros((900, 1200, 3),np.uint8)
        surface[:, :1200] = input_data['rgb_front']
        surface[:210,:280] = input_data['rgb_left']
        surface[:210, 920:1200] = input_data['rgb_right']
        surface[:210, 495:705] = input_data['rgb_center']
        surface = cv2.putText(surface, input_data['time'], (20,710), cv2.FONT_HERSHEY_TRIPLEX,0.75,(0,0,255), 1)
        surface = cv2.putText(surface, input_data['meta_control'], (20,740), cv2.FONT_HERSHEY_TRIPLEX,0.75,(0,0,255), 1)
        surface = cv2.putText(surface, input_data['waypoints'], (20,770), cv2.FONT_HERSHEY_TRIPLEX,0.75,(0,0,255), 1)
        surface = cv2.putText(surface, input_data['instruction'], (20,800), cv2.FONT_HERSHEY_TRIPLEX,0.75,(0,0,255), 1)
        surface = cv2.putText(surface, input_data['notice'], (20,830), cv2.FONT_HERSHEY_TRIPLEX,0.75,(0,0,255), 1)

        surface = cv2.putText(surface, 'Left  View', (60,245), cv2.FONT_HERSHEY_TRIPLEX,0.75,(139,69,19), 2)
        surface = cv2.putText(surface, 'Focus View', (535,245), cv2.FONT_HERSHEY_TRIPLEX,0.75,(139,69,19), 2)
        surface = cv2.putText(surface, 'Right View', (980,245), cv2.FONT_HERSHEY_TRIPLEX,0.75,(139,69,19), 2)

        surface[:210, 278:282] = [139,69,19]
        surface[:210, 493:497] = [139,69,19]
        surface[:210, 703:707] = [139,69,19]
        surface[:210, 918:922] = [139,69,19]
        surface[208:212, :280] = [139,69,19]
        surface[208:212, 920:1200] = [139,69,19]
        surface[208:212, 495:705] = [139,69,19]

        if self._enabled and self._display is not None:
            self._surface = pygame.surfarray.make_surface(surface.swapaxes(0,1))
            self._display.blit(self._surface, (0, 0))
            pygame.display.flip()
            pygame.event.get()

        return surface

    def _quit(self):
        pygame.quit()


def get_entry_point():
    return "LMDriveAgent"


class Resize2FixedSize:
    def __init__(self, size):
        self.size = size

    def __call__(self, pil_img):
        pil_img = pil_img.resize(self.size)
        return pil_img


def create_carla_rgb_transform(
    input_size, need_scale=True, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD
):

    if isinstance(input_size, (tuple, list)):
        img_size = input_size[-2:]
    else:
        img_size = input_size
    tfl = []

    if isinstance(input_size, (tuple, list)):
        input_size_num = input_size[-1]
    else:
        input_size_num = input_size

    if need_scale:
        if input_size_num == 112:
            tfl.append(Resize2FixedSize((170, 128)))
        elif input_size_num == 128:
            tfl.append(Resize2FixedSize((195, 146)))
        elif input_size_num == 224:
            tfl.append(Resize2FixedSize((341, 256)))
        elif input_size_num == 256:
            tfl.append(Resize2FixedSize((288, 288)))
        else:
            raise ValueError("Can't find proper crop size")
    tfl.append(transforms.CenterCrop(img_size))
    tfl.append(transforms.ToTensor())
    tfl.append(transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)))

    return transforms.Compose(tfl)


class LMDriveAgent(autonomous_agent.AutonomousAgent):
    def setup(self, path_to_conf_file):

        self._hic = DisplayInterface()
        self.track = autonomous_agent.Track.SENSORS
        self.step = -1
        self.wall_start = time.time()
        self.initialized = False
        self.rgb_front_transform = create_carla_rgb_transform(224)
        self.rgb_left_transform = create_carla_rgb_transform(128)
        self.rgb_right_transform = create_carla_rgb_transform(128)
        self.rgb_center_transform = create_carla_rgb_transform(128, need_scale=False)

        self.active_misleading_instruction = False
        self.remaining_misleading_frames = 0

        self.visual_feature_buffer = []

        self.config = imp.load_source("MainModel", path_to_conf_file).GlobalConfig()

        self.turn_controller = PIDController(K_P=self.config.turn_KP, K_I=self.config.turn_KI, K_D=self.config.turn_KD, n=self.config.turn_n)
        self.speed_controller = PIDController(K_P=self.config.speed_KP, K_I=self.config.speed_KI, K_D=self.config.speed_KD, n=self.config.speed_n)

        model_cls = registry.get_model_class('vicuna_drive')

        self.agent_use_notice = self.config.agent_use_notice
        self.traffic_light_notice = ''
        self.curr_notice = ''
        self.now_notice_frame_id = -1
        self.sample_rate = self.config.sample_rate * 2 # The frequency of CARLA simulation is 20Hz

        print('build model...')
        model = model_cls(preception_model=self.config.preception_model,
                          preception_model_ckpt=self.config.preception_model_ckpt,
                          llm_model=self.config.llm_model,
                          max_txt_len=64,
                          use_notice_prompt=self.config.agent_use_notice,
                          )
        self.net = model

        print('load model...')
        import gc
        gc.collect()
        ckpt = torch.load(self.config.lmdrive_ckpt, map_location='cpu')
        self.net.load_state_dict(ckpt["model"], strict=False)
        del ckpt
        gc.collect()

        # Load LoRA adapter if configured
        lora_path = getattr(self.config, 'lora_path', None)
        if lora_path is not None:
            print(f'loading LoRA adapter from {lora_path}...')
            from peft import PeftModel
            self.net.llm_model = PeftModel.from_pretrained(
                self.net.llm_model, lora_path
            )
            self.net.llm_model.eval()
            print('LoRA adapter loaded successfully')

        self.net.cuda()
        self.net.eval()
        self.softmax = torch.nn.Softmax(dim=1)
        self.prev_lidar = None
        self.prev_control = None
        self.curr_instruction = 'Drive safely.'
        self.sampled_scenarios = None
        self.instruction = ''

        # --- Runtime safety modules ---
        self.safety_enabled = SAFETY_ENABLED
        self.text_enrichment_enabled = TEXT_ENRICHMENT_ENABLED
        self.safety_supervisor = None
        self.scene_text_enricher = None
        self._text_style = TEXT_STYLE
        self._token_logger = None

        logger.info("Runtime config: safety_switch=%s, text_enrichment=%s, text_style=%s",
                     self.safety_enabled, self.text_enrichment_enabled, TEXT_STYLE)

        self.save_path = None
        if SAVE_PATH is not None:
            now = datetime.datetime.now()
            string = pathlib.Path(os.environ["ROUTES"]).stem + "_"
            string += "_".join(
                map(
                    lambda x: "%02d" % x,
                    (now.month, now.day, now.hour, now.minute, now.second),
                )
            )

            print("Data save path:", string)

            self.save_path = pathlib.Path(SAVE_PATH) / string
            self.save_path.mkdir(parents=True, exist_ok=False)
            (self.save_path / "meta").mkdir(parents=True, exist_ok=False)

    def _init(self):
        self._route_planner = RoutePlanner(5, 50.0)
        self._route_planner.set_route(self._global_plan, True)
        self._instruction_planner = InstructionPlanner(self.scenario_cofing_name, True)
        self.initialized = True
        random.seed(''.join([str(x[0]) for x in self._global_plan]))

        # Initialize Safety Supervisor if enabled
        if self.safety_enabled:
            self._setup_safety_supervisor()

        # Initialize text enrichment if enabled
        if self.text_enrichment_enabled:
            self._setup_text_enrichment()

    def _setup_safety_supervisor(self):
        """Initialize safety supervisor using CARLA's native planning."""
        try:
            from team_code.safety import get_safety_supervisor_class, get_safety_config
            from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

            world = CarlaDataProvider.get_world()
            world_map = world.get_map() if world else None
            vehicle = CarlaDataProvider.get_hero_actor()

            if not vehicle:
                logger.warning("Vehicle not available, safety supervisor disabled")
                self.safety_enabled = False
                return

            safety_cls = get_safety_supervisor_class()
            if safety_cls is None:
                logger.warning("Safety supervisor class is None (disabled by config)")
                self.safety_enabled = False
                return

            safety_config = get_safety_config()
            self.safety_supervisor = safety_cls(
                vehicle=vehicle,
                world_map=world_map,
                global_plan=getattr(self, '_global_plan_world_coord', None),
                config=safety_config,
            )
            logger.info("Safety supervisor initialized (%s)", safety_cls.__name__)
        except Exception as e:
            logger.error("Safety supervisor init failed: %s", e)
            self.safety_enabled = False

    def _setup_text_enrichment(self):
        """Initialize CARLA API text enrichment (zero VRAM cost).

        The style is selected by LMDRIVE_TEXT_STYLE: csn (default), drivevlm,
        or drivelm. All styles share the same scene-state extraction so the
        only experimental variable is the rendered text format.
        """
        try:
            from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

            style = TEXT_STYLE
            if style == "drivevlm":
                from team_code.drivevlm_text_enrichment import DriveVLMStyleTextEnrichment
                enricher_cls = DriveVLMStyleTextEnrichment
            elif style == "drivevlm_faithful":
                from team_code.drivevlm_faithful_text_enrichment import DriveVLMFaithfulTextEnrichment
                enricher_cls = DriveVLMFaithfulTextEnrichment
            elif style == "drivelm":
                from team_code.drivelm_text_enrichment import DriveLMStyleTextEnrichment
                enricher_cls = DriveLMStyleTextEnrichment
            elif style == "drivelm_faithful":
                from team_code.drivelm_faithful_text_enrichment import DriveLMFaithfulTextEnrichment
                enricher_cls = DriveLMFaithfulTextEnrichment
            else:
                from team_code.scene_text_enrichment import SceneTextEnrichment
                enricher_cls = SceneTextEnrichment
                style = "csn"
            self._text_style = style

            world = CarlaDataProvider.get_world()
            world_map = world.get_map() if world else None
            vehicle = CarlaDataProvider.get_hero_actor()

            if not vehicle:
                logger.warning("Vehicle not available, text enrichment disabled")
                self.text_enrichment_enabled = False
                return

            self.scene_text_enricher = enricher_cls(
                vehicle=vehicle,
                world=world,
                world_map=world_map,
            )
            logger.info("Text enrichment initialized: style=%s (zero VRAM)", style)

            try:
                from team_code.text_token_logger import TokenLogger
                self._token_logger = TokenLogger.get(style)
            except Exception:
                self._token_logger = None
        except Exception as e:
            logger.error("Text enrichment init failed: %s", e)
            self.text_enrichment_enabled = False

    def _get_position(self, tick_data):
        gps = tick_data["gps"]
        gps = (gps - self._route_planner.mean) * self._route_planner.scale
        return gps

    def sensors(self):
        return [
            {
                "type": "sensor.camera.rgb",
                "x": 1.3,
                "y": 0.0,
                "z": 2.3,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "width": 1200,
                "height": 900,
                "fov": 100,
                "id": "rgb_front",
            },
            {
                "type": "sensor.camera.rgb",
                "x": 1.3,
                "y": 0.0,
                "z": 2.3,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": -60.0,
                "width": 400,
                "height": 300,
                "fov": 100,
                "id": "rgb_left",
            },
            {
                "type": "sensor.camera.rgb",
                "x": 1.3,
                "y": 0.0,
                "z": 2.3,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 60.0,
                "width": 400,
                "height": 300,
                "fov": 100,
                "id": "rgb_right",
            },
            {
                "type": "sensor.camera.rgb",
                "x": -1.3,
                "y": 0.0,
                "z": 2.3,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 180.0,
                "width": 400,
                "height": 300,
                "fov": 100,
                "id": "rgb_rear",
            },
            {
                "type": "sensor.lidar.ray_cast",
                "x": 1.3,
                "y": 0.0,
                "z": 2.5,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": -90.0,
                "id": "lidar",
            },
            {
                "type": "sensor.other.imu",
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "sensor_tick": 0.05,
                "id": "imu",
            },
            {
                "type": "sensor.other.gnss",
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "sensor_tick": 0.01,
                "id": "gps",
            },
            {"type": "sensor.speedometer", "reading_frequency": 20, "id": "speed"},
        ]

    def tick(self, input_data):

        rgb_front = cv2.cvtColor(input_data["rgb_front"][1][:, :, :3], cv2.COLOR_BGR2RGB)
        rgb_left = cv2.cvtColor(input_data["rgb_left"][1][:, :, :3], cv2.COLOR_BGR2RGB)
        rgb_right = cv2.cvtColor(
            input_data["rgb_right"][1][:, :, :3], cv2.COLOR_BGR2RGB
        )
        rgb_rear = cv2.cvtColor(
            input_data["rgb_rear"][1][:, :, :3], cv2.COLOR_BGR2RGB
        )
        gps = input_data["gps"][1][:2]
        speed = input_data["speed"][1]["speed"]
        compass = input_data["imu"][1][-1]
        if (
            math.isnan(compass) == True
        ):  # It can happen that the compass sends nan for a few frames
            compass = 0.0

        result = {
            "rgb_front": rgb_front,
            "rgb_left": rgb_left,
            "rgb_right": rgb_right,
            'rgb_rear': rgb_rear,
            "gps": gps,
            "speed": speed,
            "compass": compass,
        }

        pos = self._get_position(result)

        lidar_data = input_data['lidar'][1]
        result['raw_lidar'] = lidar_data

        lidar_unprocessed = lidar_data[..., :4]
        if self.prev_lidar is not None:
            lidar_unprocessed_full = np.concatenate([lidar_unprocessed, self.prev_lidar])
        else:
            lidar_unprocessed_full = lidar_unprocessed
        self.prev_lidar = lidar_unprocessed

        lidar_processed, num_points= lidar_to_raw_features(lidar_unprocessed_full)
        result['lidar'] = lidar_processed
        result['num_points'] = num_points

        result["gps"] = pos
        next_wp, next_cmd = self._route_planner.run_step(pos)
        result["next_waypoint"] = next_wp
        result["next_command"] = next_cmd.value
        result['measurements'] = [pos[0], pos[1], compass, speed]
        result['speed'] = speed

        theta = compass + np.pi / 2
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

        local_command_point = np.array([next_wp[0] - pos[0], next_wp[1] - pos[1]])
        local_command_point = R.T.dot(local_command_point)
        result["target_point"] = local_command_point

        # Generate planned waypoints in local coords for safety supervisor fallback
        if self.safety_enabled:
            planned_local = []
            # As run for the paper. run_step() pops route points, so these
            # calls advance the agent's route planner on every tick.
            temp_pos = pos
            for _ in range(5):
                wp, _ = self._route_planner.run_step(temp_pos)
                delta = np.array([wp[0] - pos[0], wp[1] - pos[1]])
                local_wp = R.T.dot(delta)
                planned_local.append(local_wp)
                temp_pos = wp
            result["planned_waypoints_local"] = np.array(planned_local, dtype=np.float32)

        return result


    def update_and_collect(self, image_embeds):
        if 'lane' in self.curr_instruction: # change lane
            sample_rate = 1
        else:
            sample_rate = 2
        sample_rate = 2
        self.visual_feature_buffer.append(image_embeds)
        result = self.visual_feature_buffer[::self.sample_rate]
        if (len(self.visual_feature_buffer) -1) % self.sample_rate != 0:
            result.append(self.visual_feature_buffer[-1])
        return torch.stack(result, 1)

    @torch.no_grad()
    def run_step(self, input_data, timestamp):
        if not self.initialized:
            self._init()

        self.step += 1

        tick_data = self.tick(input_data)

        if self.step < 20:
            control = carla.VehicleControl()
            control.steer = float(0)
            control.throttle = float(0)
            control.brake = float(1)
            return control

        if self.step % 2 != 0 and self.step > 4:
            return self.prev_control

        velocity = tick_data["speed"]
        command = tick_data["next_command"]

        rgb_front = (
            self.rgb_front_transform(Image.fromarray(tick_data["rgb_front"]))
            .unsqueeze(0)
            .cuda()
            .float()
        )
        rgb_left = (
            self.rgb_left_transform(Image.fromarray(tick_data["rgb_left"]))
            .unsqueeze(0)
            .cuda()
            .float()
        )
        rgb_right = (
            self.rgb_right_transform(Image.fromarray(tick_data["rgb_right"]))
            .unsqueeze(0)
            .cuda()
            .float()
        )
        rgb_rear = (
            self.rgb_right_transform(Image.fromarray(tick_data["rgb_rear"]))
            .unsqueeze(0)
            .cuda()
            .float()
        )
        rgb_center = (
            self.rgb_center_transform(Image.fromarray(cv2.resize(tick_data["rgb_front"], (800, 600))))
            .unsqueeze(0)
            .cuda()
            .float()
        )

        # Base instruction from LMDrive's template planner
        base_instruction = self._instruction_planner.command2instruct(self.town_id, tick_data, self._route_planner.route)
        scenario_notice = self._instruction_planner.pos2notice(self.sampled_scenarios, tick_data)
        last_traffic_light_notice = self._instruction_planner.traffic_notice(tick_data)
        last_misleading_instruction = self._instruction_planner.command2mislead(self.town_id, tick_data)

        # CSN: the instruction stays the original template (changing it every
        # frame clears visual_feature_buffer, which destroys performance).
        # The first string returned by the renderer (the causal narration for
        # CSN) is sent through LMDrive's notice channel (64 tokens; updates do
        # not reset the buffer). The second string is used only as a fallback.
        last_instruction = base_instruction
        if self.text_enrichment_enabled and self.scene_text_enricher is not None:
            nav_cmd = int(tick_data.get("next_command", 4))
            csn_instruction, csn_notice = self.scene_text_enricher.generate(
                nav_command=nav_cmd,
                base_instruction=base_instruction,
                scenario_notice=scenario_notice,
                tick_data=tick_data,
            )
            tl = getattr(self, "_token_logger", None)
            if tl is not None:
                tl.log(csn_instruction, csn_notice)
            # Priority: scenario notice > causal chain > scene graph > traffic light
            if scenario_notice:
                last_notice = scenario_notice
            elif csn_instruction:
                last_notice = csn_instruction  # LMDrive keeps the last 64 LLaMA tokens
            elif csn_notice:
                last_notice = csn_notice
            else:
                last_notice = last_traffic_light_notice
        else:
            last_notice = scenario_notice if scenario_notice else last_traffic_light_notice

        if self.curr_instruction != last_instruction or len(self.visual_feature_buffer) > 400:
            if self.remaining_misleading_frames > 0:
                self.remaining_misleading_frames = self.remaining_misleading_frames - 1
            else:
                self.active_misleading_instruction = False
                if last_misleading_instruction!= '' and random.random() < 0.2:
                    self.curr_instruction = last_misleading_instruction
                    self.active_misleading_instruction = True
                    self.remaining_misleading_frames = 20
                else:
                    self.curr_instruction = last_instruction
                self.visual_feature_buffer = []
                self.curr_notice = ''
                self.curr_notice_frame_id = -1


        input_data = {}
        input_data["rgb_front"] = rgb_front
        input_data["rgb_left"] = rgb_left
        input_data["rgb_right"] = rgb_right
        input_data["rgb_center"] = rgb_center
        input_data["rgb_rear"] = rgb_rear
        input_data['target_point'] = torch.tensor(tick_data['target_point']).cuda().view(1,2).float()
        input_data["lidar"] = (
            torch.from_numpy(tick_data["lidar"]).float().cuda().unsqueeze(0)
        )
        input_data['num_points'] = torch.tensor([tick_data['num_points']]).cuda().unsqueeze(0)
        input_data['velocity'] = torch.tensor([tick_data['speed']]).cuda().view(1, 1).float()
        input_data['text_input'] = [self.curr_instruction]
        image_embeds = self.net.visual_encoder(input_data)
        image_embeds = self.update_and_collect(image_embeds)
        input_data['valid_frames'] = [image_embeds.size(1)]

        if last_notice != '' and last_notice != self.curr_notice:
            new_notice_flag = True
            self.curr_notice = last_notice
            self.curr_notice_frame_id = image_embeds.size(1) - 1
        else:
            new_notice_flag = False

        if self.agent_use_notice:
            input_data['notice_text'] = [self.curr_notice]
            input_data['notice_frame_id'] = [self.curr_notice_frame_id]

        with torch.cuda.amp.autocast(enabled=True):
            waypoints, is_end = self.net(input_data, inference_mode=True, image_embeds=image_embeds)

        waypoints = waypoints[-1]
        waypoints = waypoints.view(5, 2)
        end_prob = self.softmax(is_end)[-1][1] # is_end[1] means the prob of the frame is the last frame

        # --- Safety Supervisor: waypoint check ---
        waypoints_for_control = waypoints
        if self.safety_enabled and self.safety_supervisor is not None and self.step >= 40:
            tick_data['step'] = self.step
            waypoints_local = waypoints.clone()
            waypoints_local[:, 1] *= -1  # Convert from LMDrive to CARLA coordinates

            safety_is_safe, safe_waypoints_tensor, safety_deviation = self.safety_supervisor.check_waypoint_safety(
                llm_waypoints=waypoints_local,
                current_location=tick_data.get("next_waypoint"),
                tick_data=tick_data,
            )

            if not safety_is_safe:
                logger.warning("Step %d: LMDrive waypoints unsafe (dev: %.2fm), using fallback", self.step, safety_deviation)
                waypoints_for_control = safe_waypoints_tensor
            elif self.step % 100 == 0:
                logger.debug("Step %d: waypoints safe (dev: %.2fm)", self.step, safety_deviation)

        steer, throttle, brake, metadata = self.control_pid(waypoints_for_control, velocity)

        if end_prob > 0.75:
            self.visual_feature_buffer = []
            self.curr_notice = ''
            self.curr_notice_frame_id = -1

        if brake < 0.05:
            brake = 0.0
        if brake > 0.1:
            throttle = 0.0

        control = carla.VehicleControl()
        control.steer = float(steer) * 0.8
        control.throttle = float(throttle)
        control.brake = float(brake)

        # --- Safety Supervisor: control limits ---
        if self.safety_enabled and self.safety_supervisor is not None:
            control = self.safety_supervisor.check_control_safety(control)

        display_data = {}
        display_data['rgb_front'] = cv2.resize(tick_data['rgb_front'], (1200, 900))
        display_data['rgb_left'] = cv2.resize(tick_data['rgb_left'], (280, 210))
        display_data['rgb_right'] = cv2.resize(tick_data['rgb_right'], (280, 210))
        display_data['rgb_center'] = cv2.resize(tick_data['rgb_front'][330:570, 480:720], (210, 210))
        if self.active_misleading_instruction:
            display_data['instruction'] = "Instruction: [Misleading] %s" % input_data['text_input'][0]
        else:
            display_data['instruction'] = "Instruction: %s" % input_data['text_input'][0]
        display_data['time'] = 'Time: %.3f. Frames: %d. End prob: %.2f' % (timestamp, len(self.visual_feature_buffer), end_prob)
        display_data['meta_control'] = 'Throttle: %.2f. Steer: %.2f. Brake: %.2f' %(
            control.steer, control.throttle, control.brake
        )
        display_data['waypoints'] = 'Waypoints: (%.1f, %.1f), (%.1f, %.1f)' % (waypoints[0,0], -waypoints[0,1], waypoints[1,0], -waypoints[1,1])
        display_data['notice'] = "Notice: %s" % last_notice
        surface = self._hic.run_interface(display_data)
        tick_data['surface'] = surface

        if self.step % 2 != 0 and self.step > 4:
            control = self.prev_control
        else:
            self.prev_control = control

        if SAVE_PATH is not None:
            self.save(tick_data)

        return control

    def save(self, tick_data):
        frame = (self.step - 20)
        Image.fromarray(tick_data["surface"]).save(
            self.save_path / "meta" / ("%04d.jpg" % frame)
        )
        return

    def destroy(self):
        # Log intervention summary at end of route
        if self.safety_enabled and self.safety_supervisor is not None:
            if hasattr(self.safety_supervisor, 'get_intervention_summary'):
                summary = self.safety_supervisor.get_intervention_summary()
                logger.info("=== Intervention Summary ===")
                logger.info("  Total checks: %d", summary.get("total_checks", 0))
                logger.info("  Total interventions: %d", summary.get("total_interventions", 0))
                logger.info("  Intervention rate: %.4f", summary.get("intervention_rate", 0))
                logger.info("  Direction conflicts: %d", summary.get("direction_conflict_count", 0))
                logger.info("  TTC brakes: %d", summary.get("ttc_brake_count", 0))
                logger.info("  Stuck detected: %d", summary.get("stuck_detected_count", 0))
                logger.info("  Stuck recovered: %d", summary.get("stuck_recovered_count", 0))
                # Save to JSON if save path exists
                if SAVE_PATH is not None and hasattr(self, 'save_path'):
                    import json
                    summary_path = self.save_path / "intervention_summary.json"
                    try:
                        with open(summary_path, 'w') as f:
                            json.dump(summary, f, indent=2, default=str)
                        logger.info("  Saved to: %s", summary_path)
                    except Exception as e:
                        logger.warning("  Failed to save summary: %s", e)
        del self.net

    def control_pid(self, waypoints, velocity):
        '''
        Predicts vehicle control with a PID controller.
        Args:
            waypoints (tensor): predicted waypoints
            velocity (tensor): speedometer input
        '''
        assert(waypoints.size(0)==5)
        waypoints = waypoints.data.cpu().numpy()

        # flip y is (forward is negative in our waypoints)
        waypoints[:,1] *= -1
        speed = velocity

        desired_speed = np.linalg.norm(waypoints[0] - waypoints[1]) * 2.0
        brake = desired_speed < self.config.brake_speed or (speed / desired_speed) > self.config.brake_ratio

        aim = (waypoints[1] + waypoints[0]) / 2.0
        angle = np.degrees(np.pi / 2 - np.arctan2(aim[1], aim[0])) / 90
        if(speed < 0.01):
            angle = np.array(0.0) # When we don't move we don't want the angle error to accumulate in the integral
        steer = self.turn_controller.step(angle)
        steer = np.clip(steer, -1.0, 1.0)

        delta = np.clip(desired_speed - speed, 0.0, self.config.clip_delta)
        throttle = self.speed_controller.step(delta)
        throttle = np.clip(throttle, 0.0, self.config.max_throttle)
        throttle = throttle if not brake else 0.0

        metadata = {
            'speed': float(speed.astype(np.float64)),
            'steer': float(steer),
            'throttle': float(throttle),
            'brake': float(brake),
            'wp_2': tuple(waypoints[1].astype(np.float64)),
            'wp_1': tuple(waypoints[0].astype(np.float64)),
            'desired_speed': float(desired_speed.astype(np.float64)),
            'angle': float(angle.astype(np.float64)),
            'aim': tuple(aim.astype(np.float64)),
            'delta': float(delta.astype(np.float64)),
        }

        return steer, throttle, brake, metadata
