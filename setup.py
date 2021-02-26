import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
    begin_description = '\n\n# causalCannibalisation\n'
    end_description = '\n\n## Installation as a Python wheel package'

    idx_begin_description = long_description.find(begin_description) + len(begin_description)
    idx_end_description = long_description.find(end_description)
    project_description = long_description[idx_begin_description:idx_end_description]

setuptools.setup(
    name="promotionalCannibalisation",
    version="0.1.0",
    description="Methods for calculating promotional cannibalisation",
    long_description=project_description,
    long_description_content_type="text/markdown",
    author='Carlos Aguilar',
    author_email='carlos.aguilar.palacios@gmail.com',
    url="https://github.com/CarlitosDev/causalCannibalisation",
    packages=setuptools.find_packages(),
    classifiers=["Programming Language :: Python :: 3"],
    python_requires='>=3.6',
    install_requires=['numpy', 'pandas', 'scipy', 'sklearn',
    'catboost', 'networkx', 'papermill', 'jupyterlab']
)