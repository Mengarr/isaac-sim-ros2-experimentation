"""
object_randomizer.py  —  Isaac Sim Script Editor snippet
---------------------------------------------------------
Paste and run in Isaac Sim's Script Editor (Window > Script Editor).

Randomly moves the existing cube and bowl to new positions on the table
in front of the SO-101 arm.  Re-run each episode.

Coordinate frame for this robot:
    -Y  →  forward (away from robot)
    +X  →  left
    +Z  →  up
"""

import numpy as np
import omni.usd
from pxr import Gf, Usd, UsdGeom

# ── Prim paths ─────────────────────────────────────────────────────────────────
ROBOT_PRIM_PATH = "/World/so_101_with_robot_schema"
BOWL_PRIM_PATH  = "/World/shallow_bowl_blue"
CUBE_PRIM_PATH  = "/World/red_block"

# ── Spawn area (relative to robot base, metres) ────────────────────────────────
#
# Forward = -Y,  Left/right = ±X
#
#              robot base
#                  |
#       X+ ←──────┼──────→ X-
#                  |
#                 -Y (forward)
#
FORWARD_MIN = 0.15
FORWARD_MAX = 0.32
LATERAL_MIN = -0.20
LATERAL_MAX =  0.20

MIN_SEPARATION = 0.12   # bbox-centre to bbox-centre (metres)
MAX_RETRIES    = 200

# ── Helpers ────────────────────────────────────────────────────────────────────

def _get_prim(path: str):
    prim = omni.usd.get_context().get_stage().GetPrimAtPath(path)
    if not prim.IsValid():
        raise RuntimeError(f"Prim not found: {path}")
    return prim


def _get_xform_translate(path: str) -> Gf.Vec3d:
    xf  = UsdGeom.Xformable(_get_prim(path))
    ops = {op.GetOpName(): op for op in xf.GetOrderedXformOps()}
    if "xformOp:translate" in ops:
        return Gf.Vec3d(ops["xformOp:translate"].Get())
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    return cache.GetLocalToWorldTransform(_get_prim(path)).ExtractTranslation()


def _set_xform_translate(path: str, pos: Gf.Vec3d):
    xf  = UsdGeom.Xformable(_get_prim(path))
    ops = {op.GetOpName(): op for op in xf.GetOrderedXformOps()}
    if "xformOp:translate" in ops:
        ops["xformOp:translate"].Set(pos)
    else:
        xf.AddTranslateOp().Set(pos)


def _get_world_bbox(path: str):
    """Return (bbox_min, bbox_max) as Gf.Vec3d in world space."""
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
    aabb  = cache.ComputeWorldBound(_get_prim(path)).ComputeAlignedRange()
    return aabb.GetMin(), aabb.GetMax()


# ── Placement ──────────────────────────────────────────────────────────────────

def _compute_bbox_offset(path: str):
    """
    Returns (dx, dy, dz) — the vector from the xform origin to:
      - bbox centre in XY
      - bbox bottom (min Z)
    Used to back-compute the xform translate needed to put the
    bbox at a desired world position.
    """
    t          = _get_xform_translate(path)
    bmin, bmax = _get_world_bbox(path)
    cx = (bmin[0] + bmax[0]) / 2.0
    cy = (bmin[1] + bmax[1]) / 2.0
    bz = bmin[2]
    return (cx - t[0], cy - t[1], bz - t[2])


def place_object(path: str, target_x: float, target_y: float, offset):
    """
    Move prim so its bbox centre XY lands at (target_x, target_y)
    and its bbox bottom Z stays at the table (Z unchanged).
    offset = (dx, dy, dz) from _compute_bbox_offset.
    """
    _set_xform_translate(path, Gf.Vec3d(
        target_x - offset[0],
        target_y - offset[1],
        0.02 - offset[2],       # bbox bottom at Z=0.02 to avoid ground collision
    ))


# ── Sampling ───────────────────────────────────────────────────────────────────

rng = np.random.default_rng()


def sample_pair(robot_x: float, robot_y: float):
    """Sample two bbox-centre target positions separated by MIN_SEPARATION."""
    for _ in range(MAX_RETRIES):
        fwd1, lat1 = rng.uniform(FORWARD_MIN, FORWARD_MAX), rng.uniform(LATERAL_MIN, LATERAL_MAX)
        fwd2, lat2 = rng.uniform(FORWARD_MIN, FORWARD_MAX), rng.uniform(LATERAL_MIN, LATERAL_MAX)
        tx1, ty1 = robot_x + lat1, robot_y - fwd1
        tx2, ty2 = robot_x + lat2, robot_y - fwd2
        if np.hypot(tx1 - tx2, ty1 - ty2) >= MIN_SEPARATION:
            return (tx1, ty1), (tx2, ty2)

    print("[object_randomizer] Warning: MIN_SEPARATION not satisfied — using corners.")
    return (
        (robot_x + LATERAL_MIN, robot_y - FORWARD_MIN),
        (robot_x + LATERAL_MAX, robot_y - FORWARD_MAX),
    )


# ── Main ───────────────────────────────────────────────────────────────────────

robot_t  = _get_xform_translate(ROBOT_PRIM_PATH)
robot_x, robot_y = robot_t[0], robot_t[1]

# Compute bbox offsets once (xform origin → bbox centre XY / bbox bottom Z)
cube_offset = _compute_bbox_offset(CUBE_PRIM_PATH)
bowl_offset = _compute_bbox_offset(BOWL_PRIM_PATH)

(cx, cy), (bx, by) = sample_pair(robot_x, robot_y)

place_object(CUBE_PRIM_PATH, cx, cy, cube_offset)
place_object(BOWL_PRIM_PATH, bx, by, bowl_offset)

print(f"[object_randomizer] robot base      = ({robot_x:.3f}, {robot_y:.3f})")
print(f"[object_randomizer] cube bbox centre → ({cx:.3f}, {cy:.3f})")
print(f"[object_randomizer] bowl bbox centre → ({bx:.3f}, {by:.3f})")
print(f"[object_randomizer] separation       = {np.hypot(cx-bx, cy-by):.3f} m")
