
from dex_retargeting.retargeting_config import RetargetingConfig
from dex_retargeting.seq_retarget import SeqRetargeting

import json

import numpy as np

config_path='/home/hjp/ws_twist2/dex-retargeting/src/dex_retargeting/configs/teleop/inspire_hand_left_dexpilot_pico.yml'
root_path = '/home/hjp/ws_twist2/dex-retargeting/assets/robots/hands'
RetargetingConfig.set_default_urdf_dir(root_path)
retargeting = RetargetingConfig.load_from_file(config_path).build()

human_keys = [""]

# load a json file and read the data
with open("/home/hjp/ws_twist2/dex-retargeting/smplx.json", "r", encoding="utf-8") as f:
    data = json.load(f)


indices = retargeting.optimizer.target_link_human_indices

# Get finger tip positions
# Note: The order should match finger_tip_link_names in config: [thumb, index, middle, ring, pinky]
finger_tips = np.array([
    data['LeftHandThumbTip'][0],   # thumb (index 1 in robot)
    data['LeftHandIndexTip'][0],   # index (index 2 in robot)
    data['LeftHandMiddleTip'][0],  # middle (index 3 in robot)
    data['LeftHandRingTip'][0],    # ring (index 4 in robot)
    data['LeftHandLittleTip'][0]   # pinky (index 5 in robot)
])  # (5, 3)

wrist_pos = np.array([data["LeftHandWrist"][0]])  # (1, 3)

# DexPilot needs 15 vectors for 5-finger hand:
# - First 10: finger-to-finger vectors (for projection mechanism)
#   Order: 1-2, 1-3, 1-4, 1-5, 2-3, 2-4, 2-5, 3-4, 3-5, 4-5
#   Note: origin is j, task is i, so vector is task - origin = i - j
# - Last 5: wrist-to-finger vectors
#   Order: 0-1, 0-2, 0-3, 0-4, 0-5 (wrist to each finger)

# Generate finger-to-finger vectors (10 vectors)
# According to generate_link_indices: for i in [1,4], for j in [i+1,5]
# origin = j, task = i, so vector = finger_tips[i-1] - finger_tips[j-1]
finger_to_finger_vectors = []
for i in range(1, 5):  # i from 1 to 4
    for j in range(i + 1, 6):  # j from i+1 to 5
        # Vector from finger j to finger i (task - origin)
        vec = finger_tips[i-1] - finger_tips[j-1]
        finger_to_finger_vectors.append(vec)

# Generate wrist-to-finger vectors (5 vectors)
# Order: wrist to finger 1, 2, 3, 4, 5
wrist_to_finger_vectors = finger_tips - wrist_pos  # (5, 3)

# Combine all vectors: [10 finger-to-finger vectors, 5 wrist-to-finger vectors]
ref_value = np.concatenate([
    np.array(finger_to_finger_vectors),  # (10, 3)
    wrist_to_finger_vectors              # (5, 3)
], axis=0)  # Total: (15, 3)

# Check if mimic joints are being handled
has_mimic_adaptor = retargeting.optimizer.adaptor is not None
print(f"Mimic joints handling: {'YES' if has_mimic_adaptor else 'NO'}")

if has_mimic_adaptor:
    adaptor = retargeting.optimizer.adaptor
    if hasattr(adaptor, 'idx_pin2mimic'):
        print(f"  - Mimic joint indices: {adaptor.idx_pin2mimic}")
        print(f"  - Source joint indices: {adaptor.idx_pin2source}")
        print(f"  - Multipliers: {adaptor.multipliers}")
        print(f"  - Offsets: {adaptor.offsets}")
        mimic_joint_names = [retargeting.optimizer.robot.dof_joint_names[i] for i in adaptor.idx_pin2mimic]
        source_joint_names = [retargeting.optimizer.robot.dof_joint_names[i] for i in adaptor.idx_pin2source]
        print(f"  - Mimic joints: {mimic_joint_names}")
        print(f"  - Source joints: {source_joint_names}")

# retarget() returns ALL robot DOF (all joints), not just target_joint_names
robot_qpos = retargeting.retarget(ref_value)  # Shape: (robot.dof,)

print(f"\nTotal robot DOF: {len(robot_qpos)}")
print(f"Target joint names: {retargeting.optimizer.target_joint_names}")
print(f"Number of target joints: {len(retargeting.optimizer.target_joint_names)}")

# If you only want the target joints (the 6 joints being optimized):
target_qpos = robot_qpos[retargeting.optimizer.idx_pin2target]
print(f"Target joints values shape: {target_qpos.shape}")
print(f"Target joints values: {target_qpos}")

# If you want to see all joint names:
print(f"\nAll robot joint names: {retargeting.joint_names}")
print(f"All robot joint values: {robot_qpos}")

# Verify mimic joint calculations
if has_mimic_adaptor:
    adaptor = retargeting.optimizer.adaptor
    print(f"\n=== Mimic Joint Verification ===")
    for i, mimic_idx in enumerate(adaptor.idx_pin2mimic):
        source_idx = adaptor.idx_pin2source[i]
        multiplier = adaptor.multipliers[i]
        offset = adaptor.offsets[i]
        
        source_value = robot_qpos[source_idx]
        mimic_value = robot_qpos[mimic_idx]
        expected_value = source_value * multiplier + offset
        
        mimic_name = retargeting.joint_names[mimic_idx]
        source_name = retargeting.joint_names[source_idx]
        
        print(f"\n{mimic_name} (index {mimic_idx}):")
        print(f"  Source: {source_name} (index {source_idx}) = {source_value:.6f}")
        print(f"  Formula: {source_value:.6f} * {multiplier:.6f} + {offset:.6f}")
        print(f"  Expected: {expected_value:.6f}")
        print(f"  Actual:   {mimic_value:.6f}")
        print(f"  Match:    {'✓' if abs(mimic_value - expected_value) < 1e-5 else '✗'}")

# return the active joints values
active_joints_values = robot_qpos[retargeting.optimizer.idx_pin2target]
print(f"\nActive joints values: {active_joints_values}")
pass