import os
from setuptools import find_packages, setup
from glob import glob

package_name = 'acrfp'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='nimholz',
    maintainer_email='nimholz@ethz.ch',
    description='acrfp of the ACR P&S course',
    license='MIT License',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'pure_pursuit = acrfp.pure_pursuit:main',
            'mod_pp = acrfp.modified_pp:main',
            'spliner = acrfp.spliner:main',
        ],
    },
) 
