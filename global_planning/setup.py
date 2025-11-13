import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'global_planning'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Philip Dames',
    maintainer_email='pdames@temple.edu',
    description='A ROS2 package for planning global paths through an occupancy grid map',
    license='LGPL-3.0-only',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'costmap_2d_node = costmap2d.costmap_2d:main',
            'prm_node = prm.prm_node:main',
        ],
    },
)
