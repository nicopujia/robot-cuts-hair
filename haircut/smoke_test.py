"""Step 3 check: read the mock xArm7's state and wiggle joint 1."""

from dimos.manipulation.sdk import Arm
from dimos.porcelain.dimos import Dimos

app = Dimos.connect()
try:
    arm = Arm.from_app(app)
    print("info:", arm.info)
    print("joints:", arm.joints())
    print("pose:", arm.pose())
    print("presets:", list(arm.state().joint_presets.keys()))
    q0 = arm.joints()
    q = q0.copy()
    q[0] += 0.6
    print(arm.move_joints(q, speed_scale=0.5))
    print("pose after:", arm.pose())
    print(arm.move_joints(q0, speed_scale=0.5))
finally:
    app.stop()
