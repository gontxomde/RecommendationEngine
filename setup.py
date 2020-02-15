from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='RecommendationEngine',
    version='0.0.1',
    description='RecommendationEngine for films',
    long_description=readme,
    author='Lucas Mauro',
    author_email='gontxomde@gmail.com',
    url='https://github.com/gontxomde/RecommendationEngine',
    license=license,
    packages=find_packages(exclude=('docs'))
)