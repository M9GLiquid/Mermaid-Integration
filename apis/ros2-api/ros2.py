from __future__ import annotations
import argparse
import json
import re
import time
import math
import threading
import queue
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union, Optional, Callable

import numpy as np

import rclpy
from rclpy import executors
from rclpy.node import Node
from std_msgs.msg import String
from std_msgs.msg import Float32MultiArray


@dataclass
class SpiralRow:
    id: int
    row: float
    col: float
    angle: float
    certainty: float
