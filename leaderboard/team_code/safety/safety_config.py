"""
Configuration of the runtime monitors: which implementation to use and its
thresholds.
"""

import os
from enum import Enum

class SafetySupervisorType(Enum):
    """Monitor type."""
    BASIC = "basic"           # same class as ADVANCED (kept for old configs)
    ADVANCED = "advanced"     # direction-conflict monitor (paper: "Semantic Safety")
    TTC = "ttc"               # TTC-based reactive baseline (no semantic analysis)
    DISABLED = "disabled"     # no monitor

# Default type
DEFAULT_SAFETY_TYPE = SafetySupervisorType.ADVANCED

def get_safety_supervisor_type():
    """
    Return the monitor type to use.

    Order of precedence:
    1. environment variable LMDRIVE_SAFETY_TYPE
    2. environment variable LMDRIVE_SAFETY_ENABLED
    3. the default
    """
    # Explicit type
    safety_type_env = os.environ.get('LMDRIVE_SAFETY_TYPE', '').lower()
    if safety_type_env:
        for safety_type in SafetySupervisorType:
            if safety_type_env == safety_type.value:
                return safety_type
    
    # On/off switch
    safety_enabled_env = os.environ.get('LMDRIVE_SAFETY_ENABLED', 'true').lower()
    if safety_enabled_env in ['false', 'disabled', '0', 'no']:
        return SafetySupervisorType.DISABLED
    
    # Default
    return DEFAULT_SAFETY_TYPE

def get_safety_supervisor_class():
    """Return the monitor class for the configured type (None if disabled)."""
    safety_type = get_safety_supervisor_type()

    if safety_type == SafetySupervisorType.DISABLED:
        return None

    if safety_type == SafetySupervisorType.TTC:
        try:
            from .ttc_safety_supervisor import TTCSafetySupervisor
        except ImportError:
            from ttc_safety_supervisor import TTCSafetySupervisor
        return TTCSafetySupervisor

    # BASIC and ADVANCED now use the same merged SafetySupervisor
    try:
        from .safety_supervisor import SafetySupervisor
    except ImportError:
        from safety_supervisor import SafetySupervisor
    return SafetySupervisor

# Parameters
SAFETY_CONFIG = {
    # Thresholds
    'lateral_deviation_warning': 2.0,      # lateral deviation warning (m)
    'lateral_deviation_critical': 3.5,     # lateral deviation critical (m); deviation check is disabled in the monitor
    'collision_time_threshold': 3.0,       # time-to-collision (s); the TTC monitor uses its own default of 2.0 s
    'traffic_violation_threshold': 0.3,    # unused
    'overall_safety_threshold': 0.7,       # unused
    
    # Prediction (unused)
    'prediction_horizon': 5,
    'prediction_dt': 0.1,
    
    # Risk heat map (unused)
    'heatmap_grid_size': (20, 20),
    'heatmap_range': 50.0,
    
    # Control limits, applied on every tick while a monitor is enabled
    'max_steer_limit': 0.8,               # |steer| clamp
    'max_throttle_limit': 0.9,            # throttle clamp
    'safety_factor': 1.2,                 # unused
    
    # Logging
    'log_interval': 10,
    'enable_debug_output': True,
    'save_risk_heatmaps': False,
}

def get_safety_config():
    """Return the parameters, with environment-variable overrides."""
    config = SAFETY_CONFIG.copy()
    
    # Environment-variable overrides
    env_overrides = {
        'LMDRIVE_LATERAL_DEV_WARNING': 'lateral_deviation_warning',
        'LMDRIVE_LATERAL_DEV_CRITICAL': 'lateral_deviation_critical',
        'LMDRIVE_COLLISION_TIME_THRESHOLD': 'collision_time_threshold',
        'LMDRIVE_SAFETY_THRESHOLD': 'overall_safety_threshold',
        'LMDRIVE_PREDICTION_HORIZON': 'prediction_horizon',
        'LMDRIVE_MAX_STEER': 'max_steer_limit',
        'LMDRIVE_MAX_THROTTLE': 'max_throttle_limit',
        'LMDRIVE_DEBUG_OUTPUT': 'enable_debug_output',
    }
    
    for env_var, config_key in env_overrides.items():
        env_value = os.environ.get(env_var)
        if env_value is not None:
            try:
                if config_key in ['enable_debug_output', 'save_risk_heatmaps']:
                    config[config_key] = env_value.lower() in ['true', '1', 'yes', 'on']
                elif config_key == 'prediction_horizon':
                    config[config_key] = int(env_value)
                else:
                    config[config_key] = float(env_value)
            except (ValueError, TypeError):
                print(f"Warning: cannot parse environment variable {env_var}={env_value}")
    
    return config
