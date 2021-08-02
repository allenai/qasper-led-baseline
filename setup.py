from setuptools import setup, find_packages

setup(name='qasper_led_baseline',
      version='0.1',
      description='Longformer Encoder Decoder model for Qasper',
      author='Pradeep Dasigi',
      author_email='pradeepd@allenai.org',
      packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
      license='Apache',
     )
