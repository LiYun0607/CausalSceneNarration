"""Safety supervision module for LMDrive."""

from .safety_supervisor import SafetySupervisor
from .ttc_safety_supervisor import TTCSafetySupervisor
from .safety_config import get_safety_supervisor_class, get_safety_config, SAFETY_CONFIG

# Backward compatibility aliases
AdvancedSafetySupervisor = SafetySupervisor

__all__ = [
    'SafetySupervisor',
    'TTCSafetySupervisor',
    'AdvancedSafetySupervisor',
    'get_safety_supervisor_class',
    'get_safety_config',
    'SAFETY_CONFIG',
]
