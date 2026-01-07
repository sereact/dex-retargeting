
from dex_retargeting.retargeting_config import RetargetingConfig
from dex_retargeting.seq_retarget import SeqRetargeting

import json
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import sapien
from sapien.asset import create_dome_envmap
from sapien.utils import Viewer

config_path='/home/hjp/ws_twist2/dex-retargeting/src/dex_retargeting/configs/teleop/inspire_hand_right_dexpilot_pico.yml'
root_path = '/home/hjp/ws_twist2/dex-retargeting/assets/robots/hands'
RetargetingConfig.set_default_urdf_dir(root_path)
retargeting = RetargetingConfig.load_from_file(config_path).build()

human_keys = [""]

# load a json file and read the data
with open("/home/hjp/ws_twist2/TWIST2/smplx.json", "r", encoding="utf-8") as f:
    data = json.load(f)


indices = retargeting.optimizer.target_link_human_indices

# Get finger tip positions
# Note: The order should match finger_tip_link_names in config: [thumb, index, middle, ring, pinky]
finger_tips = np.array([
    data['RightHandThumbTip'][0],   # thumb (index 1 in robot)
    data['RightHandIndexTip'][0],   # index (index 2 in robot)
    data['RightHandMiddleTip'][0],  # middle (index 3 in robot)
    data['RightHandRingTip'][0],    # ring (index 4 in robot)
    data['RightHandLittleTip'][0]   # pinky (index 5 in robot)
])  # (5, 3)

quat_finger_tips = np.array([data['RightHandThumbTip'][1], data['RightHandIndexTip'][1], data['RightHandMiddleTip'][1], data['RightHandRingTip'][1], data['RightHandLittleTip'][1]])
quat_wrist = np.array([data["RightHandWrist"][1]])

wrist_pos = np.array([data["RightHandWrist"][0]])  # (1, 3)

# ============================================================================
# Transform finger_tips to wrist coordinate frame
# ============================================================================
# quat format: w x y z
# Convert wrist quaternion to rotation object
quat_wrist_xyzw = quat_wrist[0]  # Extract the quaternion [w, x, y, z]
R_wrist = R.from_quat(quat_wrist_xyzw[[1, 2, 3, 0]])  # scipy uses [x, y, z, w] format
R_wrist_inv = R_wrist.inv()  # Inverse rotation matrix

# Transform finger_tips positions to wrist coordinate frame
# Step 1: Translate to wrist origin
finger_tips_translated = finger_tips - wrist_pos  # (5, 3)
# Step 2: Rotate to wrist coordinate frame
finger_tips = R_wrist_inv.apply(finger_tips_translated)  # (5, 3)
finger_tips -= np.array([0,0,0.08]) #NOTE: hack offset

# Transform finger_tips rotations to wrist coordinate frame
# For each finger tip quaternion, compute: q_wrist_inv * q_finger
quat_finger_tips_transformed = []
for i in range(len(quat_finger_tips)):
    quat_finger_xyzw = quat_finger_tips[i]  # [w, x, y, z]
    R_finger = R.from_quat(quat_finger_xyzw[[1, 2, 3, 0]])  # scipy uses [x, y, z, w]
    
    # Compute relative rotation: R_wrist_inv * R_finger
    R_relative = R_wrist_inv * R_finger
    quat_relative_xyzw = R_relative.as_quat()  # Returns [x, y, z, w]
    quat_relative_wxyz = np.array([quat_relative_xyzw[3], quat_relative_xyzw[0], 
                                    quat_relative_xyzw[1], quat_relative_xyzw[2]])  # Convert to [w, x, y, z]
    quat_finger_tips_transformed.append(quat_relative_wxyz)

quat_finger_tips = np.array(quat_finger_tips_transformed)  # (5, 4)

# Wrist is now at origin with identity rotation
wrist_pos = np.zeros(3)
quat_wrist = np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion [w, x, y, z]

# convert from smplx to urdf coordinate frame
smplx2urdf = np.array([[0,-1, 0],[-1,0,0],[0,0,-1]])
finger_tips = finger_tips @ smplx2urdf
wrist_pos = wrist_pos @ smplx2urdf

# Save human data for sapien visualization (after coordinate transformation)
human_wrist_pos = wrist_pos.copy()
human_finger_tips = finger_tips.copy()
human_quat_wrist = quat_wrist.copy()  # [w, x, y, z]
human_quat_finger_tips = quat_finger_tips.copy()  # (5, 4) [w, x, y, z]

# ============================================================================
# Open3D Visualization: wrist and finger tips with coordinate frame
# ============================================================================
# Create point cloud for wrist and finger tips
all_points = np.vstack([wrist_pos, finger_tips])  # (6, 3)
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(all_points)

# Set colors: wrist in red, finger tips in blue
colors = np.array([
    [1.0, 0.0, 0.0],  # Red for wrist
    [0.0, 0.0, 1.0],  # Blue for thumb
    [0.0, 0.0, 1.0],  # Blue for index
    [0.0, 0.0, 1.0],  # Blue for middle
    [0.0, 0.0, 1.0],  # Blue for ring
    [0.0, 0.0, 1.0],  # Blue for pinky
])
point_cloud.colors = o3d.utility.Vector3dVector(colors)

# Create coordinate frame at wrist position
# The coordinate frame will be shown as three arrows: X (red), Y (green), Z (blue)
coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=0.05,  # Size of the coordinate frame
    origin=wrist_pos  # Position at wrist
)

# Create visualization
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="Wrist and Finger Tips Visualization")
vis.add_geometry(point_cloud)
vis.add_geometry(coordinate_frame)

# Set point size for better visibility
render_option = vis.get_render_option()
render_option.point_size = 10.0

# Run visualization
print("\n显示可视化窗口...")
print("按 'Q' 或关闭窗口以继续")
vis.run()
vis.destroy_window()

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

# ============================================================================
# Compute actual finger positions from solved joint angles and compare with ref_value
# ============================================================================
print(f"\n{'='*80}")
print("=== 求解后的值与 ref_value 的差距 ===")
print(f"{'='*80}")

# Compute forward kinematics with solved joint angles
robot = retargeting.optimizer.robot
robot.compute_forward_kinematics(robot_qpos)

# Get link indices for wrist and finger tips
computed_link_indices = retargeting.optimizer.computed_link_indices
computed_link_names = retargeting.optimizer.computed_link_names

# Extract positions for all computed links
actual_link_poses = [robot.get_link_pose(link_id) for link_id in computed_link_indices]
actual_link_positions_dict = {name: pose[:3, 3] for name, pose in zip(computed_link_names, actual_link_poses)}

# Identify wrist and finger tips from computed_link_names
# For DexPilot: computed_link_names contains wrist + finger tips
# Finger tips typically have "_tip" suffix, wrist is usually "base" or similar
wrist_link_name = None
finger_tip_link_names = []

for name in computed_link_names:
    if "_tip" in name.lower() or "tip" in name.lower():
        finger_tip_link_names.append(name)
    else:
        # This should be the wrist
        wrist_link_name = name

# Sort finger tips to match expected order: thumb, index, middle, ring, pinky
finger_tip_order = ["thumb", "index", "middle", "ring", "pinky"]
finger_tip_link_names_sorted = []
for tip_keyword in finger_tip_order:
    for tip_name in finger_tip_link_names:
        if tip_keyword in tip_name.lower():
            finger_tip_link_names_sorted.append(tip_name)
            break

# If sorting failed, use original order
if len(finger_tip_link_names_sorted) != len(finger_tip_link_names):
    finger_tip_link_names_sorted = finger_tip_link_names

# Get wrist position
if wrist_link_name and wrist_link_name in actual_link_positions_dict:
    actual_wrist_pos = np.array([actual_link_positions_dict[wrist_link_name]])  # (1, 3)
else:
    # Fallback: use first link
    actual_wrist_pos = np.array([actual_link_poses[0][:3, 3]])  # (1, 3)
    wrist_link_name = computed_link_names[0]

# Get finger tip positions in sorted order
actual_finger_tips = np.array([actual_link_positions_dict[name] for name in finger_tip_link_names_sorted])  # (5, 3)

print(f"\n实际手指尖位置 (相对于机器人坐标系):")
for i, name in enumerate(finger_tip_link_names_sorted):
    print(f"  {name}: [{actual_finger_tips[i][0]:.6f}, {actual_finger_tips[i][1]:.6f}, {actual_finger_tips[i][2]:.6f}]")

# ============================================================================
# Open3D Visualization: Compare reference (input) vs actual (retargeting result)
# ============================================================================
# Prepare reference points (original input after coordinate transform)
ref_wrist_pos = wrist_pos  # Already transformed to wrist coordinate frame
ref_finger_tips = finger_tips  # Already transformed to wrist coordinate frame

# Prepare actual points (from retargeting)
actual_wrist_pos_flat = actual_wrist_pos[0]  # Flatten from (1, 3) to (3,)

# Combine all points: [ref_wrist, ref_tips(5), actual_wrist, actual_tips(5)]
all_points = np.vstack([
    ref_wrist_pos,           # Reference wrist (1 point)
    ref_finger_tips,         # Reference finger tips (5 points)
    actual_wrist_pos_flat.reshape(1, -1),  # Actual wrist (1 point)
    actual_finger_tips       # Actual finger tips (5 points)
])  # Total: 12 points

# Create point cloud
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(all_points)

# Set colors:
# Reference wrist: Red
# Reference finger tips: Blue
# Actual wrist: Green
# Actual finger tips: Yellow
colors = np.array([
    [1.0, 0.0, 0.0],  # Red for reference wrist
    [0.0, 0.0, 1.0],  # Blue for reference thumb
    [0.0, 0.0, 1.0],  # Blue for reference index
    [0.0, 0.0, 1.0],  # Blue for reference middle
    [0.0, 0.0, 1.0],  # Blue for reference ring
    [0.0, 0.0, 1.0],  # Blue for reference pinky
    [0.0, 1.0, 0.0],  # Green for actual wrist
    [0.0, 1.0, 0.0],  # Yellow for actual thumb
    [0.0, 1.0, 0.0],  # Yellow for actual index
    [0.0, 1.0, 0.0],  # Yellow for actual middle
    [0.0, 1.0, 0.0],  # Yellow for actual ring
    [0.0, 1.0, 0.0],  # Yellow for actual pinky
])
point_cloud.colors = o3d.utility.Vector3dVector(colors)

# Create coordinate frames
# Reference coordinate frame at reference wrist
ref_coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=0.05,
    origin=ref_wrist_pos
)

# Actual coordinate frame at actual wrist
actual_coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=0.05,
    origin=actual_wrist_pos_flat
)

# Create lines connecting reference and actual points for better visualization
# Connect reference wrist to actual wrist
wrist_line = o3d.geometry.LineSet()
wrist_line.points = o3d.utility.Vector3dVector([ref_wrist_pos, actual_wrist_pos_flat])
wrist_line.lines = o3d.utility.Vector2iVector([[0, 1]])
wrist_line.colors = o3d.utility.Vector3dVector([[0.5, 0.5, 0.5]])  # Gray

# Connect each reference finger tip to corresponding actual finger tip
finger_lines = []
for i in range(len(ref_finger_tips)):
    line = o3d.geometry.LineSet()
    line.points = o3d.utility.Vector3dVector([ref_finger_tips[i], actual_finger_tips[i]])
    line.lines = o3d.utility.Vector2iVector([[0, 1]])
    line.colors = o3d.utility.Vector3dVector([[0.5, 0.5, 0.5]])  # Gray
    finger_lines.append(line)

# Create visualization
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="Retargeting Comparison: Reference (Red/Blue) vs Actual (Green/Yellow)")
vis.add_geometry(point_cloud)
vis.add_geometry(ref_coordinate_frame)
vis.add_geometry(actual_coordinate_frame)
vis.add_geometry(wrist_line)
for line in finger_lines:
    vis.add_geometry(line)

# Set point size for better visibility
render_option = vis.get_render_option()
render_option.point_size = 10.0

# Run visualization
print("\n显示retargeting对比可视化窗口...")
print("红色/蓝色: 参考值 (输入)")
print("绿色/黄色: 实际值 (retargeting结果)")
print("灰色线条: 连接对应的参考点和实际点")
print("按 'Q' 或关闭窗口以继续")
vis.run()
vis.destroy_window()

# Compute actual vectors using the same order as ref_value
# First 10: finger-to-finger vectors (1-2, 1-3, 1-4, 1-5, 2-3, 2-4, 2-5, 3-4, 3-5, 4-5)
actual_finger_to_finger_vectors = []
for i in range(1, 5):  # i from 1 to 4
    for j in range(i + 1, 6):  # j from i+1 to 5
        vec = actual_finger_tips[i-1] - actual_finger_tips[j-1]
        actual_finger_to_finger_vectors.append(vec)

# Last 5: wrist-to-finger vectors
actual_wrist_to_finger_vectors = actual_finger_tips - actual_wrist_pos  # (5, 3)

# Combine all actual vectors
actual_value = np.concatenate([
    np.array(actual_finger_to_finger_vectors),  # (10, 3)
    actual_wrist_to_finger_vectors              # (5, 3)
], axis=0)  # Total: (15, 3)

# Compute differences
diff = actual_value - ref_value
diff_norm = np.linalg.norm(diff, axis=1)  # (15,)
ref_norm = np.linalg.norm(ref_value, axis=1)  # (15,)
relative_error = diff_norm / (ref_norm + 1e-8)  # Avoid division by zero

print(f"\n向量对比 (前10个为手指间向量, 后5个为手腕到手指向量):")
print(f"{'Index':<6} {'Type':<25} {'Ref Norm':<12} {'Actual Norm':<12} {'Diff Norm':<12} {'Relative Error':<15}")
print("-" * 90)

# Finger-to-finger vectors (indices 0-9)
finger_names = ["thumb", "index", "middle", "ring", "pinky"]
for idx in range(10):
    # Find which fingers this vector connects
    i, j = None, None
    count = 0
    for fi in range(1, 5):
        for fj in range(fi + 1, 6):
            if count == idx:
                i, j = fi, fj
                break
            count += 1
        if i is not None:
            break
    
    vec_type = f"{finger_names[i-1]}-{finger_names[j-1]}"
    print(f"{idx:<6} {vec_type:<25} {ref_norm[idx]:<12.6f} {np.linalg.norm(actual_value[idx]):<12.6f} {diff_norm[idx]:<12.6f} {relative_error[idx]*100:<14.4f}%")

# Wrist-to-finger vectors (indices 10-14)
for idx in range(10, 15):
    finger_idx = idx - 10
    vec_type = f"wrist-{finger_names[finger_idx]}"
    print(f"{idx:<6} {vec_type:<25} {ref_norm[idx]:<12.6f} {np.linalg.norm(actual_value[idx]):<12.6f} {diff_norm[idx]:<12.6f} {relative_error[idx]*100:<14.4f}%")

# Summary statistics
print(f"\n{'='*80}")
print("=== 统计摘要 ===")
print(f"{'='*80}")
print(f"平均绝对误差 (L2 norm): {np.mean(diff_norm):.6f} m")
print(f"最大绝对误差: {np.max(diff_norm):.6f} m")
print(f"最小绝对误差: {np.min(diff_norm):.6f} m")
print(f"平均相对误差: {np.mean(relative_error)*100:.4f}%")
print(f"最大相对误差: {np.max(relative_error)*100:.4f}%")
print(f"最小相对误差: {np.min(relative_error)*100:.4f}%")
print(f"\n总误差 (所有向量的L2范数): {np.linalg.norm(diff):.6f} m")
print(f"参考值总范数: {np.linalg.norm(ref_value):.6f} m")
print(f"实际值总范数: {np.linalg.norm(actual_value):.6f} m")

# Detailed vector comparison for first few vectors
print(f"\n{'='*80}")
print("=== 前3个向量的详细对比 ===")
print(f"{'='*80}")
for idx in range(min(3, len(ref_value))):
    print(f"\n向量 {idx}:")
    print(f"  参考值: [{ref_value[idx][0]:.6f}, {ref_value[idx][1]:.6f}, {ref_value[idx][2]:.6f}]")
    print(f"  实际值: [{actual_value[idx][0]:.6f}, {actual_value[idx][1]:.6f}, {actual_value[idx][2]:.6f}]")
    print(f"  差值:   [{diff[idx][0]:.6f}, {diff[idx][1]:.6f}, {diff[idx][2]:.6f}]")
    print(f"  误差范数: {diff_norm[idx]:.6f} m, 相对误差: {relative_error[idx]*100:.4f}%")

# ============================================================================
# Sapien Visualization: Render robot hand with retargeted joint angles
# ============================================================================
print(f"\n{'='*80}")
print("=== Sapien 可视化 ===")
print(f"{'='*80}")

# Setup sapien renderer
sapien.render.set_viewer_shader_dir("default")
sapien.render.set_camera_shader_dir("default")

# Create scene
scene = sapien.Scene()

# Ground
render_mat = sapien.render.RenderMaterial()
render_mat.base_color = [0.06, 0.08, 0.12, 1]
render_mat.metallic = 0.0
render_mat.roughness = 0.9
render_mat.specular = 0.8
scene.add_ground(-0.2, render_material=render_mat, render_half_size=[1000, 1000])

# Lighting
scene.add_directional_light(np.array([1, 1, -1]), np.array([3, 3, 3]))
scene.add_point_light(np.array([2, 2, 2]), np.array([2, 2, 2]), shadow=False)
scene.add_point_light(np.array([2, -2, 2]), np.array([2, 2, 2]), shadow=False)
scene.set_environment_map(
    create_dome_envmap(sky_color=[0.2, 0.2, 0.2], ground_color=[0.2, 0.2, 0.2])
)

# Camera
cam = scene.add_camera(
    name="Robot Hand Camera", width=600, height=600, fovy=1, near=0.1, far=10
)
cam.set_local_pose(sapien.Pose([0.50, 0, 0.0], [0, 0, 0, -1]))

# Load robot
loader = scene.create_urdf_loader()
config = RetargetingConfig.load_from_file(config_path)
filepath = Path(config.urdf_path)
robot_name = filepath.stem
loader.load_multiple_collisions_from_file = True

# Set scale based on robot type
if "inspire" in robot_name:
    loader.scale = 1.0  # Default scale for inspire hand

# Load GLB version if available
# if "glb" not in robot_name:
    # filepath = str(filepath).replace(".urdf", "_glb.urdf")
# else:
filepath = str(filepath)
robot = loader.load(filepath)

# Map joint names from retargeting to sapien
sapien_joint_names = [joint.get_name() for joint in robot.get_active_joints()]
retargeting_joint_names = retargeting.joint_names
retargeting_to_sapien = np.array(
    [retargeting_joint_names.index(name) for name in sapien_joint_names]
).astype(int)

# Set robot joint angles
robot.set_qpos(np.array(robot_qpos)[retargeting_to_sapien])

# ============================================================================
# Add human data visualization: wrist pose and finger tips
# ============================================================================
# Create material for human data visualization
human_material = sapien.render.RenderMaterial()
human_material.base_color = [1.0, 0.0, 0.0, 1.0]  # Red for wrist
human_material.metallic = 0.0
human_material.roughness = 0.5

finger_material = sapien.render.RenderMaterial()
finger_material.base_color = [0.0, 0.0, 1.0, 1.0]  # Blue for finger tips
finger_material.metallic = 0.0
finger_material.roughness = 0.5

# Add wrist sphere
wrist_radius = 0.015
wrist_builder = scene.create_actor_builder()
wrist_builder.add_sphere_visual(radius=wrist_radius, material=human_material)
wrist_actor = wrist_builder.build(name="human_wrist")
wrist_actor.set_pose(sapien.Pose(human_wrist_pos))

# Add finger tip spheres
finger_radius = 0.01
finger_actors = []
finger_names = ["thumb", "index", "middle", "ring", "pinky"]
for i, (tip_pos, tip_name) in enumerate(zip(human_finger_tips, finger_names)):
    finger_builder = scene.create_actor_builder()
    finger_builder.add_sphere_visual(radius=finger_radius, material=finger_material)
    finger_actor = finger_builder.build(name=f"human_finger_{tip_name}")
    finger_actor.set_pose(sapien.Pose(tip_pos))
    finger_actors.append(finger_actor)

# Add coordinate frame at wrist to show orientation
# Convert quaternion from [w, x, y, z] to sapien format [x, y, z, w]
quat_wrist_sapien = np.array([human_quat_wrist[1], human_quat_wrist[2], 
                               human_quat_wrist[3], human_quat_wrist[0]])
wrist_pose = sapien.Pose(human_wrist_pos, quat_wrist_sapien)

# Get rotation matrix from quaternion
R_wrist_matrix = wrist_pose.to_transformation_matrix()[:3, :3]

# Create coordinate frame using capsules (X: red, Y: green, Z: blue)
frame_size = 0.03
axis_radius = 0.002

# Helper function to create axis arrow
def create_axis_arrow(axis_direction, color, name):
    """Create a capsule aligned with the given axis direction"""
    axis_direction = axis_direction / (np.linalg.norm(axis_direction) + 1e-8)
    axis_end = human_wrist_pos + axis_direction * frame_size
    axis_mid = (human_wrist_pos + axis_end) / 2
    
    # Default capsule is along Z axis [0, 0, 1]
    z_axis = np.array([0, 0, 1])
    if np.abs(np.dot(axis_direction, z_axis)) < 0.99:  # Not parallel
        rotation_axis = np.cross(z_axis, axis_direction)
        rotation_axis = rotation_axis / (np.linalg.norm(rotation_axis) + 1e-8)
        rotation_angle = np.arccos(np.clip(np.dot(z_axis, axis_direction), -1, 1))
        axis_quat = R.from_rotvec(rotation_axis * rotation_angle).as_quat()  # [x, y, z, w]
    else:
        axis_quat = np.array([0, 0, 0, 1])
    
    builder = scene.create_actor_builder()
    builder.add_capsule_visual(radius=axis_radius, half_length=frame_size/2,
                                material=sapien.render.RenderMaterial(base_color=color))
    actor = builder.build(name=name)
    actor.set_pose(sapien.Pose(axis_mid, axis_quat))
    return actor

# Create axis arrows
x_arrow = create_axis_arrow(R_wrist_matrix[:, 0], [1, 0, 0, 1], "wrist_frame_x")
y_arrow = create_axis_arrow(R_wrist_matrix[:, 1], [0, 1, 0, 1], "wrist_frame_y")
z_arrow = create_axis_arrow(R_wrist_matrix[:, 2], [0, 0, 1, 1], "wrist_frame_z")

# Add lines connecting wrist to finger tips
line_material = sapien.render.RenderMaterial()
line_material.base_color = [0.5, 0.5, 0.5, 0.5]  # Gray, semi-transparent
line_material.metallic = 0.0
line_material.roughness = 0.5

for i, tip_pos in enumerate(human_finger_tips):
    # Create a line from wrist to finger tip using a thin capsule
    direction = tip_pos - human_wrist_pos
    length = np.linalg.norm(direction)
    if length > 0:
        direction = direction / length
        mid_point = (human_wrist_pos + tip_pos) / 2
        
        # Create rotation to align capsule with direction
        # Default capsule is along Z axis
        z_axis = np.array([0, 0, 1])
        if np.abs(np.dot(direction, z_axis)) < 0.99:  # Not parallel
            rotation_axis = np.cross(z_axis, direction)
            rotation_axis = rotation_axis / (np.linalg.norm(rotation_axis) + 1e-8)
            rotation_angle = np.arccos(np.clip(np.dot(z_axis, direction), -1, 1))
            quat_line = R.from_rotvec(rotation_axis * rotation_angle).as_quat()  # [x, y, z, w]
        else:
            quat_line = np.array([0, 0, 0, 1])
        
        line_builder = scene.create_actor_builder()
        line_builder.add_capsule_visual(radius=0.001, half_length=length/2, material=line_material)
        line_actor = line_builder.build(name=f"wrist_to_finger_{i}")
        line_actor.set_pose(sapien.Pose(mid_point, quat_line))

print("已添加 human data 可视化:")
print(f"  - 手腕位置: {human_wrist_pos}")
print(f"  - 手指尖位置数量: {len(human_finger_tips)}")

# Create viewer
viewer = Viewer()
viewer.set_scene(scene)
viewer.control_window.show_origin_frame = False
viewer.control_window.move_speed = 0.01
viewer.control_window.toggle_camera_lines(False)
viewer.set_camera_pose(cam.get_local_pose())

# Run visualization
print("\n显示 Sapien 可视化窗口...")
print("按 'Q' 或关闭窗口以退出")
while not viewer.closed:
    viewer.render()

scene = None
