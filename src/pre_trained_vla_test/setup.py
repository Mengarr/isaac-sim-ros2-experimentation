from setuptools import find_packages, setup

package_name = 'pre_trained_vla_test'

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
    maintainer='rohan',
    maintainer_email='rohanmeagher@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'pi0_inference = pre_trained_vla_test.pi0_inference:main',
            'pi0_inference_server = pre_trained_vla_test.pi0_inference_server:main',
            'pi0_inference_broker = pre_trained_vla_test.pi0_inference_broker:main',
            'act_inference = pre_trained_vla_test.act_inference:main',
        ],
    },
)
