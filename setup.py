"""
Setup configuration for the Multimodal AI System package.

This file enables the project to be installed as a pip package using:
    pip install -e .

The package can then be imported and used from anywhere in the Python environment.
"""

from setuptools import setup, find_packages
import os

# Read the README file for long description
def read_readme():
    """Read the README.md file for the long description."""
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "A comprehensive multimodal AI system combining computer vision and natural language processing."

# Read requirements from requirements.txt
def read_requirements():
    """Read the requirements.txt file for dependencies."""
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    requirements = []
    if os.path.exists(requirements_path):
        with open(requirements_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                # Skip empty lines and comments
                if line and not line.startswith('#'):
                    requirements.append(line)
    return requirements

setup(
    # Basic package information
    name="multimodal-ai-system",
    version="1.0.0",
    author="Multimodal AI Team",
    author_email="team@multimodal-ai.com",
    description="A comprehensive multimodal AI system combining OpenCV, YOLOv8, and BLIP",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/multimodal-ai-system",

    # Package discovery
    packages=find_packages(include=['src', 'src.*']),
    package_dir={'': '.'},

    # Dependencies
    install_requires=read_requirements(),

    # Python version requirement
    python_requires=">=3.8",

    # Package classification
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],

    # Keywords for discoverability
    keywords=[
        "computer vision", "natural language processing", "multimodal AI",
        "object detection", "image captioning", "OCR", "YOLO", "BLIP",
        "OpenCV", "transformers", "deep learning", "machine learning"
    ],

    # Entry points for command-line interfaces
    entry_points={
        'console_scripts': [
            'multimodal-ai-web=src.web_app:main',
        ],
    },

    # Include additional files
    include_package_data=True,
    package_data={
        'src': ['*.py'],
        '': ['requirements.txt', 'README.md', 'data/README.md'],
    },

    # Extra dependencies for development
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'pytest-cov>=4.0.0',
            'black>=22.0.0',
            'flake8>=5.0.0',
            'mypy>=1.0.0',
        ],
        'gpu': [
            'torch>=2.0.0+cu118',
            'torchvision>=0.15.0+cu118',
            'torchaudio>=2.0.0+cu118',
        ],
        'full': [
            'pytest>=7.0.0',
            'pytest-cov>=4.0.0',
            'black>=22.0.0',
            'flake8>=5.0.0',
            'mypy>=1.0.0',
            'torch>=2.0.0+cu118',
            'torchvision>=0.15.0+cu118',
            'torchaudio>=2.0.0+cu118',
        ]
    },

    # Project URLs
    project_urls={
        "Bug Reports": "https://github.com/your-org/multimodal-ai-system/issues",
        "Source": "https://github.com/your-org/multimodal-ai-system",
        "Documentation": "https://github.com/your-org/multimodal-ai-system/wiki",
    },

    # Zip safety
    zip_safe=False,
)
