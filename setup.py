from setuptools import setup, find_packages

setup(
    name="multimodal-disentangled-recommender",
    version="0.1.0",
    description="Multimodal Sequential Recommendation with Disentangled Representation and Causal Inference",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "scipy>=1.11.0",
        "torchvision>=0.15.0",
        "transformers>=4.30.0",
        "einops>=0.7.0",
    ],
    extras_require={
        "quantum": ["qiskit>=0.45.0", "pennylane>=0.32.0"],
        "dev": ["pytest>=7.4.0", "black>=23.7.0", "flake8>=6.1.0"],
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
