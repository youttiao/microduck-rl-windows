"""Head orientation -> duck joint angles, solved against the real MJCF.

Why an IK instead of a formula
-------------------------------
The duck's head is four hinges: `neck_pitch`, `head_pitch`, `head_yaw`,
`head_roll`. Their axes are all "0 0 1" in their own bodies' local frames,
and the `neck_pitch` body is mounted 180 degrees off, so the joints do NOT
compose into a simple Euler chain: `neck_pitch > 0` tips the head BACK while
`head_pitch > 0` nods the chin DOWN. Hand-writing the composition gets the
signs wrong in a way that is easy to miss and hard to debug.

So: build the real model and solve numerically. The head's "look" frame is
the `head_camera` site (x = forward, y = left, z = up) -- the frame the robot
actually has. At HOME it coincides with the trunk frame.

Solver
------
Damped least squares on the orientation error. For a site rigidly attached to
a body, the site's angular velocity is the body's, and for hinge joints the
body's angular velocity is the sum of axis_i * dq_i, so the angular Jacobian
is just the joint axes in world coordinates.

The model is loaded once; each solve is a handful of forward passes, which is
milliseconds -- fast enough to retarget a whole motion offline.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

# The four head joints, in the order the solver works with them.
HEAD_JOINTS = ("neck_pitch", "head_pitch", "head_yaw", "head_roll")

# The duck's standing pose (microduck_constants.HOME_FRAME). The legs matter
# for the trunk's absolute attitude, so they are needed to know where the
# trunk actually sits at rest.
HOME_HEAD = {"neck_pitch": 0.3491, "head_pitch": 0.3491, "head_yaw": 0.0, "head_roll": 0.0}

HOME_LEGS = {
    "left_hip_yaw": 0.0, "left_hip_roll": -0.0873, "left_hip_pitch": -0.4579,
    "left_knee": -0.0049, "left_ankle": 0.4530,
    "right_hip_yaw": 0.0, "right_hip_roll": 0.0873, "right_hip_pitch": 0.4579,
    "right_knee": 0.0049, "right_ankle": -0.4530,
}


class DuckHeadIK:
    """Solve the four head joints for a desired head orientation.

    Parameters
    ----------
    xml : path to the duck's MJCF (the ground-contact model, which is what
        the expressive env uses).
    look_site : the site whose frame defines "the head is looking this way".
    """

    def __init__(self, xml: str | Path, look_site: str = "head_camera",
                 home: dict[str, float] | None = None):
        self.model = mujoco.MjModel.from_xml_path(str(xml))
        self.data = mujoco.MjData(self.model)
        self.look_site = look_site

        self.jid = {
            name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in HEAD_JOINTS
        }
        if any(j < 0 for j in self.jid.values()):
            missing = [n for n, j in self.jid.items() if j < 0]
            raise ValueError(f"model has no {missing} joint(s)")

        # qpos is addressed by jnt_qposadr, NOT by joint id: the free joint at
        # the root occupies qpos[0:7], so joint id 7 is qpos[13]. Writing
        # qpos[jid] corrupts the root quaternion instead of a joint.
        self.qadr = {
            name: int(self.model.jnt_qposadr[self.jid[name]])
            for name in HEAD_JOINTS
        }

        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, look_site)
        if sid < 0:
            raise ValueError(f"model has no site {look_site!r}")
        self.sid = sid

        # Joint limits straight from the model, so the solver never proposes
        # something the hardware cannot do.
        self.lo = np.array([self.model.jnt_range[self.jid[n], 0] for n in HEAD_JOINTS])
        self.hi = np.array([self.model.jnt_range[self.jid[n], 1] for n in HEAD_JOINTS])
        # The same limits keyed by the expressive env's channel names, so a
        # caller can check "is this reference inside the joint's range?".
        self.lo_names = list(HEAD_JOINTS)

        # The body each joint rotates, so we can get its world axis.
        self.jnt_body = np.array([self.model.jnt_bodyid[self.jid[n]] for n in HEAD_JOINTS])

        home = home or HOME_HEAD
        self.q_home = np.array([home[n] for n in HEAD_JOINTS])
        self._set(self.q_home)
        mujoco.mj_forward(self.model, self.data)
        # The head's frame at HOME: the neutral pose the human's neutral head
        # is mapped onto.
        self.R_home = self._site_R().copy()

        # The duck at its full standing pose: where the trunk actually sits
        # and where the head looks, relative to the trunk. The env measures
        # the trunk's ABSOLUTE roll and pitch, so a retargeted reference has
        # to start from these numbers, not from zero.
        self.trunk_rpy_home = self._trunk_rpy_at_home()
        self.R_rel_home = self._head_rel_to_trunk_at_home()

    # -- internals ---------------------------------------------------------

    def _set(self, q: np.ndarray) -> None:
        for name, v in zip(HEAD_JOINTS, q):
            self.data.qpos[self.qadr[name]] = v

    def _set_full_home(self) -> None:
        """Put the whole robot in its standing pose (legs included)."""
        for name, v in {**HOME_LEGS, **HOME_HEAD}.items():
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid >= 0:
                self.data.qpos[int(self.model.jnt_qposadr[jid])] = v
        # The free joint's quaternion: identity, i.e. the trunk upright.
        root = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
        if root >= 0:
            adr = int(self.model.jnt_qposadr[root])
            self.data.qpos[adr + 3] = 1.0

    def _trunk_rpy_at_home(self) -> np.ndarray:
        """(roll, pitch, yaw) of the trunk at the full standing pose."""
        self._set_full_home()
        mujoco.mj_forward(self.model, self.data)
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
        R = self.data.xmat[bid].reshape(3, 3)
        roll = float(np.arctan2(R[2, 1], R[2, 2]))
        pitch = float(np.arcsin(np.clip(-R[2, 0], -1.0, 1.0)))
        yaw = float(np.arctan2(R[1, 0], R[0, 0]))
        return np.array([roll, pitch, yaw])

    def _head_rel_to_trunk_at_home(self) -> np.ndarray:
        """The head's frame relative to the trunk, at the standing pose."""
        self._set_full_home()
        mujoco.mj_forward(self.model, self.data)
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
        R_t = self.data.xmat[bid].reshape(3, 3)
        return R_t.T @ self._site_R()

    def _site_R(self) -> np.ndarray:
        return self.data.site_xmat[self.sid].reshape(3, 3).copy()

    def _axes_world(self) -> np.ndarray:
        """(3, 4) world axes of the four head joints."""
        out = np.zeros((3, len(HEAD_JOINTS)))
        for k, b in enumerate(self.jnt_body):
            R = self.data.xmat[b].reshape(3, 3)
            a = self.model.jnt_axis[self.jid[HEAD_JOINTS[k]]]
            out[:, k] = R @ a
        return out

    # -- public ------------------------------------------------------------

    def head_orientation(self, q: np.ndarray) -> np.ndarray:
        """World rotation of the look frame for joint values `q`."""
        self._set(np.asarray(q, dtype=float))
        mujoco.mj_forward(self.model, self.data)
        return self._site_R()

    def solve(self, target_R: np.ndarray, q_init: np.ndarray | None = None,
              iters: int = 60, damping: float = 1e-3) -> tuple[np.ndarray, float]:
        """Joint values that put the look frame closest to `target_R`.

        Returns (q, residual_radians). The residual is the geodesic angle
        between the achieved and the target orientation.
        """
        q = np.array(self.q_home if q_init is None else q_init, dtype=float)
        q = np.clip(q, self.lo, self.hi)
        lam = damping
        for it in range(iters):
            R = self.head_orientation(q)
            err = Rotation.from_matrix(target_R @ R.T).as_rotvec()
            if np.linalg.norm(err) < 1e-4:
                break
            J = self._axes_world()
            # Damped least squares:  dq = J^T (J J^T + lam^2 I)^-1 err
            JJt = J @ J.T + (lam ** 2) * np.eye(3)
            dq = J.T @ np.linalg.solve(JJt, err)
            # Damp harder when we are far away, so the first steps do not
            # fling the joints to their limits.
            lam = max(1e-4, lam * (0.9 if np.linalg.norm(err) < 0.3 else 1.05))
            q = np.clip(q + dq, self.lo, self.hi)
        R = self.head_orientation(q)
        residual = float(np.linalg.norm(Rotation.from_matrix(target_R @ R.T).as_rotvec()))
        return q, residual

    def solve_series(self, targets: np.ndarray, q_init: np.ndarray | None = None):
        """Solve a sequence of orientations, warm-starting each from the last.

        Returns (qs (N, 4), residuals (N,)).
        """
        n = len(targets)
        qs = np.zeros((n, len(HEAD_JOINTS)))
        res = np.zeros(n)
        q = None if q_init is None else np.array(q_init, dtype=float)
        for i in range(n):
            q, r = self.solve(targets[i], q_init=q)
            qs[i] = q
            res[i] = r
        return qs, res
