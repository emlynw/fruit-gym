import numpy as np

PANDA_HOME = np.array([0.0, -1.6, 0.0, -2.54, -0.05, 2.49, 0.822], dtype=np.float32)
GRIPPER_HOME = np.array([0.0141, 0.0141], dtype=np.float32)
GRIPPER_MIN = 0
GRIPPER_MAX = 0.004
PANDA_XYZ = np.array([0.1, 0, 0.8], dtype=np.float32)
CARTESIAN_BOUNDS = np.array([[0.05, -0.2, 0.6], [0.55, 0.2, 0.95]], dtype=np.float32)
ROTATION_BOUNDS = np.array([[-np.pi/3, -np.pi/6, -np.pi/10],[np.pi/3, np.pi/6, np.pi/10]], dtype=np.float32)
default_obj_pos = np.array([0.42, 0, 0.85])
gripper_sleep = 0.6
grasp_threshold = 0.333
ripe_mats = {"r1", "r2", "r3"}
