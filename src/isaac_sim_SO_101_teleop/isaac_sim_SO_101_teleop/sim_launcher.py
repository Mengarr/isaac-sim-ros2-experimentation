"""
sim_launcher.py
---------------
Standalone Isaac Sim 5.x script for the SO-101 arm.

Run with Isaac Sim's Python interpreter:
    ./python.sh /path/to/isaac_sim_bridge/isaac_sim_bridge/sim_launcher.py

Requirements:
  - ROS2 sourced in the terminal before running Isaac Sim
  - isaacsim.ros2.bridge extension enabled (handled below)
  - sensor_msgs available in sourced ROS2 workspace

PS4 DualShock mapping
---------------------
Axes:
  0  → Left stick X   → shoulder_pan   (velocity)
  1  → Left stick Y   → shoulder_lift  (velocity)
  3  → Right stick X  → elbow_flex     (velocity)
  4  → Right stick Y  → wrist_flex     (velocity)
  2  → L2 trigger     → gripper close  (1.0=idle, -1.0=full press)
  5  → R2 trigger     → gripper open   (1.0=idle, -1.0=full press)

Buttons:
  4  → L1             → wrist_roll CW  (fixed velocity while held)
  5  → R1             → wrist_roll CCW (fixed velocity while held)

Note: PS4 triggers (axes 2, 5) default to 1.0 and travel to -1.0 when
fully pressed. We remap this to a [0.0, 1.0] drive fraction.
"""

# ── 1. Bootstrap SimulationApp before any omni imports ────────────────────────
from isaacsim import SimulationApp

simulation_app = SimulationApp({
    "headless": False,
    "width": 1280,
    "height": 720,
    "renderer": "RaytracedLighting",
})

# ── 2. Standard omni / Isaac imports (after SimulationApp) ────────────────────
import carb
import numpy as np
import threading
import time

import omni.usd
import omni.graph.core as og
from isaacsim.core.api import World
from isaacsim.core.prims import Articulation
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.extensions import enable_extension
from isaacsim.storage.native import get_assets_root_path
from pxr import Gf

# ── 3. Enable ROS2 bridge extension ───────────────────────────────────────────
enable_extension("isaacsim.ros2.bridge")
simulation_app.update()  # allow extension to initialise

# ── 4. ROS2 imports (only valid after bridge extension is enabled) ─────────────
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy

# ── 5. Constants ──────────────────────────────────────────────────────────────

ROBOT_USD_SUBPATH = "/Isaac/Robots/RobotStudio/so101_new_calib/so101_new_calib.usd"
ROBOT_PRIM_PATH   = "/World/SO101"

# Joint names as defined in the SO-101 URDF (Isaac Sim preserves these).
# A runtime check will print actual names if these are wrong — see below.
JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

# Joint limits from URDF (radians) — used to clamp velocity integration
JOINT_LIMITS = {
    "shoulder_pan":  (-1.91986,  1.91986),
    "shoulder_lift": (-1.74533,  1.74533),
    "elbow_flex":    (-1.69000,  1.69000),
    "wrist_flex":    (-1.65806,  1.65806),
    "wrist_roll":    (-2.74385,  2.84121),
    "gripper":       (-0.17453,  1.74533),
}

# Maximum joint velocity (rad/s) for thumbstick axes at full deflection
MAX_VELOCITY = {
    "shoulder_pan":  1.0,
    "shoulder_lift": 1.0,
    "elbow_flex":    1.0,
    "wrist_flex":    1.0,
    "wrist_roll":    0.8,   # used for fixed L1/R1 speed too
    "gripper":       0.8,
}

# Simulation timestep (seconds) — used to integrate velocity → position
SIM_DT = 1.0 / 60.0

# ── 6. Joy → Joint velocity mapper ───────────────────────────────────────────

class JoyState:
    """Thread-safe container for the latest Joy message data."""

    def __init__(self):
        self._lock = threading.Lock()
        self._axes    = [0.0] * 8
        self._buttons = [0]   * 12

    def update(self, axes, buttons):
        with self._lock:
            self._axes    = list(axes)
            self._buttons = list(buttons)

    def get(self):
        with self._lock:
            return list(self._axes), list(self._buttons)


def joy_to_velocities(axes, buttons):
    """
    Map PS4 input to a dict of {joint_name: velocity_rad_s}.

    Returns velocities; caller integrates into positions each sim step.
    """
    vels = {j: 0.0 for j in JOINT_NAMES}

    # ── Thumbsticks → velocity for 4 arm joints ────────────────────────────
    # Axes default at ~0.0 with slight deadzone noise; apply a small deadzone.
    def deadzone(v, dz=0.05):
        return v if abs(v) > dz else 0.0

    # Left stick X  (axis 0) → shoulder_pan
    # Note: axis convention — positive axis = positive joint direction.
    # Flip sign here if the arm moves the wrong way.
    vels["shoulder_pan"]  = deadzone(axes[0]) * MAX_VELOCITY["shoulder_pan"]

    # Left stick Y  (axis 1) → shoulder_lift
    # Stick Y is typically inverted on PS4 (up = negative); un-invert here.
    vels["shoulder_lift"] = -deadzone(axes[1]) * MAX_VELOCITY["shoulder_lift"]

    # Right stick X (axis 3) → elbow_flex
    vels["elbow_flex"]    = deadzone(axes[3]) * MAX_VELOCITY["elbow_flex"]

    # Right stick Y (axis 4) → wrist_flex (un-inverted)
    vels["wrist_flex"]    = -deadzone(axes[4]) * MAX_VELOCITY["wrist_flex"]

    # ── L1 / R1 buttons → wrist_roll ──────────────────────────────────────
    # Button 4 = L1 → CW  (+velocity)
    # Button 5 = R1 → CCW (-velocity)
    wrist_roll_vel = 0.0
    if len(buttons) > 4 and buttons[4]:
        wrist_roll_vel += MAX_VELOCITY["wrist_roll"]
    if len(buttons) > 5 and buttons[5]:
        wrist_roll_vel -= MAX_VELOCITY["wrist_roll"]
    vels["wrist_roll"] = wrist_roll_vel

    # ── L2 / R2 triggers → gripper ────────────────────────────────────────
    # Triggers default 1.0, range to -1.0 at full press.
    # Remap to [0, 1] fraction: fraction = (1.0 - raw) / 2.0
    l2_fraction = (1.0 - axes[2]) / 2.0  # L2 → close gripper (+velocity)
    r2_fraction = (1.0 - axes[5]) / 2.0  # R2 → open gripper  (-velocity)
    vels["gripper"] = (l2_fraction - r2_fraction) * MAX_VELOCITY["gripper"]

    return vels


# ── 7. ROS2 subscriber node ───────────────────────────────────────────────────

class JoySubscriber(Node):

    def __init__(self, joy_state: JoyState):
        super().__init__("isaac_joy_subscriber")
        self._joy_state = joy_state
        self.create_subscription(Joy, "/joy", self._cb, 10)
        self.get_logger().info("Subscribed to /joy")

    def _cb(self, msg: Joy):
        self._joy_state.update(msg.axes, msg.buttons)


# ── 8. Scene builder ──────────────────────────────────────────────────────────

def build_scene(world: World):
    """
    Programmatically create a minimal scene: ground plane + SO-101 arm.
    Everything here can later be overridden by loading your own .usda file.
    """
    stage = omni.usd.get_context().get_stage()

    # Ground plane
    world.scene.add_default_ground_plane()

    # Lighting — dome light so the arm is visible
    from pxr import UsdLux
    dome_path = "/World/DomeLight"
    dome_light = UsdLux.DomeLight.Define(stage, dome_path)
    dome_light.CreateIntensityAttr(500.0)

    # SO-101 arm — referenced from Isaac Sim asset library
    assets_root = get_assets_root_path()
    if assets_root is None:
        raise RuntimeError(
            "Could not find Isaac Sim assets root. "
            "Is Nucleus / local assets configured?"
        )

    robot_usd = assets_root + ROBOT_USD_SUBPATH
    carb.log_info(f"Loading robot from: {robot_usd}")

    add_reference_to_stage(usd_path=robot_usd, prim_path=ROBOT_PRIM_PATH)

    # Adjust spawn position if the arm clips the ground (edit Z as needed).
    # Set via attribute because the USD already defines xformOp:translate.
    stage.GetPrimAtPath(ROBOT_PRIM_PATH).GetAttribute("xformOp:translate").Set(
        Gf.Vec3d(0.0, 0.0, 0.0)
    )

    return world


# ── 9. Main loop ──────────────────────────────────────────────────────────────

def main():
    # --- World ---
    world = World(physics_dt=SIM_DT, rendering_dt=SIM_DT, stage_units_in_meters=1.0)
    build_scene(world)
    world.reset()

    # --- Articulation ---
    robot = Articulation(prim_path=ROBOT_PRIM_PATH, name="so101")
    world.scene.add(robot)
    world.reset()  # second reset initialises articulation

    # Print actual joint names from USD so you can verify / fix JOINT_NAMES
    actual_names = robot.dof_names
    carb.log_warn(f"[SO101] Actual DOF names from USD: {actual_names}")
    print(f"\n>>> SO-101 DOF names from USD: {actual_names}\n")

    # Build a position array we'll update each step (start at zeros / home)
    n_dof = robot.num_dof
    joint_positions = np.zeros(n_dof)

    # Map joint name → dof index for fast lookup
    name_to_idx = {name: i for i, name in enumerate(actual_names)}

    # Warn if any expected joint name is missing
    for jname in JOINT_NAMES:
        if jname not in name_to_idx:
            carb.log_warn(
                f"[SO101] Expected joint '{jname}' not found in USD. "
                f"Available: {actual_names}. Check JOINT_NAMES constant."
            )

    # --- ROS2 ---
    rclpy.init()
    joy_state = JoyState()
    joy_node  = JoySubscriber(joy_state)
    ros_thread = threading.Thread(target=rclpy.spin, args=(joy_node,), daemon=True)
    ros_thread.start()

    print(">>> Simulation running. Move PS4 sticks to control the arm.")
    print(">>> Close the Isaac Sim window to exit.\n")

    # --- Sim loop ---
    while simulation_app.is_running():
        world.step(render=True)

        if not world.is_playing():
            # Auto-play on first loop
            world.play()
            continue

        axes, buttons = joy_state.get()
        vels = joy_to_velocities(axes, buttons)

        # Integrate velocity → position for each controlled joint
        for jname, vel in vels.items():
            idx = name_to_idx.get(jname)
            if idx is None:
                continue

            lo, hi = JOINT_LIMITS.get(jname, (-np.pi, np.pi))
            joint_positions[idx] = float(
                np.clip(joint_positions[idx] + vel * SIM_DT, lo, hi)
            )

        # Send position targets to articulation controller
        from isaacsim.core.utils.types import ArticulationAction
        action = ArticulationAction(joint_positions=joint_positions.copy())
        robot.apply_action(action)

    # --- Cleanup ---
    joy_node.destroy_node()
    rclpy.shutdown()
    simulation_app.close()


if __name__ == "__main__":
    main()