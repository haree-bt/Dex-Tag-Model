import re

# Read base ur5e SDF
with open('/tmp/ur5e.sdf', 'r') as f:
    sdf_content = f.read()

# Replace model name
sdf_content = sdf_content.replace("<model name='ur5e'>", "<model name='ur5e_manipulator'>")

# Remove fixed base joint to world if present (so we can place it on table in world SDF)
# In ur5e.sdf, base_joint has parent 'world'. We can change parent to '__link__' or keep it,
# or better: we attach base_link to the table or ground.

gripper_and_plugins = """
    <!-- ======================================================= -->
    <!-- DEXTEROUS 3-FINGER ADAPTIVE END-EFFECTOR               -->
    <!-- ======================================================= -->
    <!-- Gripper Palm -->
    <joint name='palm_joint' type='fixed'>
      <pose relative_to='wrist_3_link'>0 0.08 0 0 1.5707963 0</pose>
      <parent>wrist_3_link</parent>
      <child>gripper_palm</child>
    </joint>
    <link name='gripper_palm'>
      <pose relative_to='palm_joint'>0 0 0 0 0 0</pose>
      <inertial>
        <mass>0.6</mass>
        <inertia>
          <ixx>0.0005</ixx><iyy>0.0005</iyy><izz>0.0005</izz>
          <ixy>0</ixy><ixz>0</ixz><iyz>0</iyz>
        </inertia>
      </inertial>
      <visual name='palm_vis'>
        <geometry><box><size>0.06 0.08 0.035</size></box></geometry>
        <material>
          <ambient>0.12 0.12 0.14 1</ambient>
          <diffuse>0.15 0.15 0.18 1</diffuse>
          <specular>0.7 0.7 0.7 1</specular>
        </material>
      </visual>
      <collision name='palm_col'>
        <geometry><box><size>0.06 0.08 0.035</size></box></geometry>
      </collision>
    </link>

    <!-- Finger 1: Thumb (Left Opposing Digit) -->
    <joint name='finger_thumb_joint' type='prismatic'>
      <pose relative_to='gripper_palm'>0 -0.025 0.04 0 0 0</pose>
      <parent>gripper_palm</parent>
      <child>finger_thumb</child>
      <axis>
        <xyz>0 1 0</xyz>
        <limit><lower>-0.03</lower><upper>0.02</upper></limit>
        <dynamics><damping>10</damping><friction>1.0</friction></dynamics>
      </axis>
    </joint>
    <link name='finger_thumb'>
      <pose relative_to='finger_thumb_joint'>0 0 0 0 0 0</pose>
      <inertial>
        <mass>0.08</mass>
        <inertia><ixx>0.00005</ixx><iyy>0.00005</iyy><izz>0.00005</izz></inertia>
      </inertial>
      <visual name='thumb_vis'>
        <pose>0 0 0.025 0 0 0</pose>
        <geometry><cylinder><radius>0.008</radius><length>0.05</length></cylinder></geometry>
        <material><ambient>0.9 0.45 0.1 1</ambient><diffuse>0.95 0.5 0.1 1</diffuse></material>
      </visual>
      <collision name='thumb_col'>
        <pose>0 0 0.025 0 0 0</pose>
        <geometry><cylinder><radius>0.008</radius><length>0.05</length></cylinder></geometry>
        <surface>
          <friction><ode><mu>2.0</mu><mu2>2.0</mu2></ode></friction>
        </surface>
      </collision>
    </link>

    <!-- Finger 2: Index (Right Upper Digit) -->
    <joint name='finger_index_joint' type='prismatic'>
      <pose relative_to='gripper_palm'>-0.015 0.025 0.04 0 0 0</pose>
      <parent>gripper_palm</parent>
      <child>finger_index</child>
      <axis>
        <xyz>0 -1 0</xyz>
        <limit><lower>-0.03</lower><upper>0.02</upper></limit>
        <dynamics><damping>10</damping><friction>1.0</friction></dynamics>
      </axis>
    </joint>
    <link name='finger_index'>
      <pose relative_to='finger_index_joint'>0 0 0 0 0 0</pose>
      <inertial>
        <mass>0.08</mass>
        <inertia><ixx>0.00005</ixx><iyy>0.00005</iyy><izz>0.00005</izz></inertia>
      </inertial>
      <visual name='index_vis'>
        <pose>0 0 0.025 0 0 0</pose>
        <geometry><cylinder><radius>0.008</radius><length>0.05</length></cylinder></geometry>
        <material><ambient>0.9 0.45 0.1 1</ambient><diffuse>0.95 0.5 0.1 1</diffuse></material>
      </visual>
      <collision name='index_col'>
        <pose>0 0 0.025 0 0 0</pose>
        <geometry><cylinder><radius>0.008</radius><length>0.05</length></cylinder></geometry>
        <surface>
          <friction><ode><mu>2.0</mu><mu2>2.0</mu2></ode></friction>
        </surface>
      </collision>
    </link>

    <!-- Finger 3: Middle (Right Lower Digit) -->
    <joint name='finger_middle_joint' type='prismatic'>
      <pose relative_to='gripper_palm'>0.015 0.025 0.04 0 0 0</pose>
      <parent>gripper_palm</parent>
      <child>finger_middle</child>
      <axis>
        <xyz>0 -1 0</xyz>
        <limit><lower>-0.03</lower><upper>0.02</upper></limit>
        <dynamics><damping>10</damping><friction>1.0</friction></dynamics>
      </axis>
    </joint>
    <link name='finger_middle'>
      <pose relative_to='finger_middle_joint'>0 0 0 0 0 0</pose>
      <inertial>
        <mass>0.08</mass>
        <inertia><ixx>0.00005</ixx><iyy>0.00005</iyy><izz>0.00005</izz></inertia>
      </inertial>
      <visual name='middle_vis'>
        <pose>0 0 0.025 0 0 0</pose>
        <geometry><cylinder><radius>0.008</radius><length>0.05</length></cylinder></geometry>
        <material><ambient>0.9 0.45 0.1 1</ambient><diffuse>0.95 0.5 0.1 1</diffuse></material>
      </visual>
      <collision name='middle_col'>
        <pose>0 0 0.025 0 0 0</pose>
        <geometry><cylinder><radius>0.008</radius><length>0.05</length></cylinder></geometry>
        <surface>
          <friction><ode><mu>2.0</mu><mu2>2.0</mu2></ode></friction>
        </surface>
      </collision>
    </link>

    <!-- ======================================================= -->
    <!-- GAZEBO HARMONIC JOINT POSITION CONTROLLER PLUGINS       -->
    <!-- ======================================================= -->
    <plugin filename="gz-sim-joint-position-controller-system" name="gz::sim::systems::JointPositionController">
      <joint_name>shoulder_pan_joint</joint_name>
      <topic>/model/ur5e_manipulator/joint/shoulder_pan_joint/cmd_pos</topic>
      <p_gain>5000</p_gain><i_gain>10</i_gain><d_gain>200</d_gain><i_max>50</i_max><i_min>-50</i_min>
    </plugin>
    <plugin filename="gz-sim-joint-position-controller-system" name="gz::sim::systems::JointPositionController">
      <joint_name>shoulder_lift_joint</joint_name>
      <topic>/model/ur5e_manipulator/joint/shoulder_lift_joint/cmd_pos</topic>
      <p_gain>6000</p_gain><i_gain>20</i_gain><d_gain>300</d_gain><i_max>100</i_max><i_min>-100</i_min>
    </plugin>
    <plugin filename="gz-sim-joint-position-controller-system" name="gz::sim::systems::JointPositionController">
      <joint_name>elbow_joint</joint_name>
      <topic>/model/ur5e_manipulator/joint/elbow_joint/cmd_pos</topic>
      <p_gain>4000</p_gain><i_gain>10</i_gain><d_gain>150</d_gain><i_max>50</i_max><i_min>-50</i_min>
    </plugin>
    <plugin filename="gz-sim-joint-position-controller-system" name="gz::sim::systems::JointPositionController">
      <joint_name>wrist_1_joint</joint_name>
      <topic>/model/ur5e_manipulator/joint/wrist_1_joint/cmd_pos</topic>
      <p_gain>1500</p_gain><i_gain>5</i_gain><d_gain>40</d_gain><i_max>20</i_max><i_min>-20</i_min>
    </plugin>
    <plugin filename="gz-sim-joint-position-controller-system" name="gz::sim::systems::JointPositionController">
      <joint_name>wrist_2_joint</joint_name>
      <topic>/model/ur5e_manipulator/joint/wrist_2_joint/cmd_pos</topic>
      <p_gain>1500</p_gain><i_gain>5</i_gain><d_gain>40</d_gain><i_max>20</i_max><i_min>-20</i_min>
    </plugin>
    <plugin filename="gz-sim-joint-position-controller-system" name="gz::sim::systems::JointPositionController">
      <joint_name>wrist_3_joint</joint_name>
      <topic>/model/ur5e_manipulator/joint/wrist_3_joint/cmd_pos</topic>
      <p_gain>1000</p_gain><i_gain>5</i_gain><d_gain>20</d_gain><i_max>10</i_max><i_min>-10</i_min>
    </plugin>
    <plugin filename="gz-sim-joint-position-controller-system" name="gz::sim::systems::JointPositionController">
      <joint_name>finger_thumb_joint</joint_name>
      <topic>/model/ur5e_manipulator/joint/finger_thumb_joint/cmd_pos</topic>
      <p_gain>800</p_gain><d_gain>10</d_gain>
    </plugin>
    <plugin filename="gz-sim-joint-position-controller-system" name="gz::sim::systems::JointPositionController">
      <joint_name>finger_index_joint</joint_name>
      <topic>/model/ur5e_manipulator/joint/finger_index_joint/cmd_pos</topic>
      <p_gain>800</p_gain><d_gain>10</d_gain>
    </plugin>
    <plugin filename="gz-sim-joint-position-controller-system" name="gz::sim::systems::JointPositionController">
      <joint_name>finger_middle_joint</joint_name>
      <topic>/model/ur5e_manipulator/joint/finger_middle_joint/cmd_pos</topic>
      <p_gain>800</p_gain><d_gain>10</d_gain>
    </plugin>

    <!-- Joint State Publisher for ROS 2 -->
    <plugin filename="gz-sim-joint-state-publisher-system" name="gz::sim::systems::JointStatePublisher">
      <topic>/joint_states</topic>
    </plugin>

    <!-- Detachable Joint System for Firm Grasp Transport -->
    <plugin filename="gz-sim-detachable-joint-system" name="gz::sim::systems::DetachableJoint">
      <parent_link>gripper_palm</parent_link>
      <child_model>industrial_target</child_model>
      <child_link>tool_link</child_link>
      <topic>/model/ur5e_manipulator/detachable_joint/attach</topic>
      <suppress_child_warning>true</suppress_child_warning>
    </plugin>
"""

# Insert before </model>
pos = sdf_content.rfind("</model>")
final_sdf = sdf_content[:pos] + gripper_and_plugins + sdf_content[pos:]

target_path = '/mnt/c/Users/hareh/.gemini/antigravity/scratch/dex_tag_net/dex_tag_ros/models/ur5e_manipulator/model.sdf'
with open(target_path, 'w') as f:
    f.write(final_sdf)

print(f"Successfully generated {target_path}!")
