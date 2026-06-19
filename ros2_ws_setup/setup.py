from setuptools import find_packages, setup

package_name = 'robot_classifier'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='student',
    maintainer_email='student@mq.edu.au',
    description='COMP8430 Phase 3',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # Demo 1 — camera only, no movement
            'robot_classifier = robot_classifier.robot_classifier_node:main',
            # Demo 2 + 3 — scan mode and target mode
            'robot_demo = robot_classifier.camera_classifier_action_node:main',
        ],
    },
)