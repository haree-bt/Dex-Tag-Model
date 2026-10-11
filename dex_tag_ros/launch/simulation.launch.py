import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    pkg_dex_tag = get_package_share_directory('dex_tag_ros')
    pkg_ros_gz_sim = get_package_share_directory('ros_gz_sim')

    world_sdf = os.path.join(pkg_dex_tag, 'worlds', 'tabletop_dexterous.sdf')
    bridge_config = os.path.join(pkg_dex_tag, 'config', 'ros_gz_bridge.yaml')
    models_dir = os.path.join(pkg_dex_tag, 'models')
    ur_share = '/opt/ros/jazzy/share'

    curr_res = os.environ.get('GZ_SIM_RESOURCE_PATH', '')
    res_path = f"{models_dir}:{ur_share}:{curr_res}" if curr_res else f"{models_dir}:{ur_share}"

    set_env_res = SetEnvironmentVariable('GZ_SIM_RESOURCE_PATH', res_path)
    set_env_sdf = SetEnvironmentVariable('SDF_PATH', res_path)

    # 1. Gazebo Sim Launch
    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_ros_gz_sim, 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={'gz_args': f'-r {world_sdf}'}.items()
    )

    # 2. Topic Bridge
    ros_gz_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        parameters=[{'config_file': bridge_config}],
        output='screen'
    )

    # 3. Static TF (World -> Camera)
    static_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['0.4', '0.0', '1.05', '0.0', '1.5707963', '0.0', 'world', 'camera_link'],
        output='screen'
    )

    # 4. Dex-TAG Perception Node
    dex_node = Node(
        package='dex_tag_ros',
        executable='dex_tag_node',
        name='dex_tag_node',
        output='screen'
    )

    # 5. Dexterous Manipulation Supervisor
    commander_node = Node(
        package='dex_tag_ros',
        executable='dex_commander',
        name='dex_commander',
        output='screen'
    )

    # 6. RViz2 Visualizer (3D Markers & Real-Time Grasp Heatmap View)
    rviz_config = os.path.join(pkg_dex_tag, 'config', 'dex_tag.rviz')
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],
        output='screen'
    )

    return LaunchDescription([
        set_env_res,
        set_env_sdf,
        gz_sim,
        ros_gz_bridge,
        static_tf,
        dex_node,
        commander_node,
        rviz_node
    ])
