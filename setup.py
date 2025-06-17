from setuptools import setup, find_packages

setup(
    name="skuast-tree-analysis",
    version="2.0.0",
    packages=find_packages(),
    install_requires=[
        'fastapi==0.68.1',
        'uvicorn==0.15.0',
        'pandas==1.3.3',
        'numpy==1.21.2',
        'scikit-learn==0.24.2',
        'joblib==1.0.1',
        'python-multipart==0.0.5',
        'pydantic==1.8.2',
        'opencv-python==4.5.3.56',
        'pillow==8.3.2',
        'flask==2.0.1',
        'gunicorn==20.1.0',
        'python-dotenv==0.19.0',
        'requests==2.26.0',
        'scipy==1.7.1',
        'matplotlib==3.4.3',
        'seaborn==0.11.2'
    ],
    python_requires='>=3.7',
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
) 