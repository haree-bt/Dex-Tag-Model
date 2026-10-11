import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'dex_tag_ros'

setup(
    name=package_name,
    version='0.2.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*.sdf')),
        (os.path.join('share', package_name, 'config'), glob('config/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Hareharan B T',
    maintainer_email='hareharanbt22@gmail.com',
    description='Dex-TAG: Dexterous Task-Aware Grasping in ROS 2 Jazzy & Gazebo Harmonic',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'dex_tag_node = dex_tag_ros.dex_tag_node:main',
            'dex_commander = dex_tag_ros.dex_commander:main',
        ],
    },
)
