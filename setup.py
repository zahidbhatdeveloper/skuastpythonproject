from setuptools import setup, find_packages

setup(
    name="skuast-tree-analysis",
    version="2.0.0",
    packages=find_packages(),
    install_requires=[
        'fastapi==0.88.0',
        'uvicorn==0.20.0',
        'pandas==1.4.4',
        'numpy==1.23.5',
        'scikit-learn==1.1.3',
        'joblib==1.1.0',
        'python-multipart==0.0.5',
        'pydantic==1.10.2',
        'opencv-python-headless==4.6.0.66',
        'pillow==9.2.0',
        'flask==2.2.3',
        'gunicorn==20.1.0',
        'python-dotenv==0.21.0',
        'requests==2.28.1',
        'scipy==1.9.3',
        'matplotlib==3.6.2',
        'seaborn==0.12.1'
    ],
    python_requires='>=3.9,<3.10',
    setup_requires=['setuptools>=42', 'wheel'],
    author="SKUAST",
    author_email="your.email@example.com",
    description="Tree Analysis System with ML-based predictions",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
) #comment