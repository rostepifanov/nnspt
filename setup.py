import setuptools

from pathlib import Path

INSTALL_REQUIRES = ['torch>=1.7', 'timm==0.9.2']

def get_version():
    locals_ = dict()

    with open(Path(__file__).parent / 'nnspt' / '__version__.py') as f:
        exec(f.read(), globals(), locals_)
        return locals_['__version__']

def get_long_description():
    with open(Path(__file__).parent / 'README.md', encoding='utf-8') as f:
        return f.read()

setuptools.setup(
    name='nnspt',
    version=get_version(),
    description='',
    long_description=get_long_description(),
    long_description_content_type='text/markdown',
    author='Rostislav Epifanov',
    author_email='rostepifanov@gmail.com',
    license='MIT',
    url='https://github.com/rostepifanov/nnspt',
    packages=setuptools.find_packages(exclude=['tests']),
    python_requires='>=3.7',
    install_requires=INSTALL_REQUIRES,
    extras_require={'tests': ['pytest']},
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: MIT License',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ]
)
