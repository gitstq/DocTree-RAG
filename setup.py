import os
from setuptools import setup, find_packages


def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as f:
            return f.read()
    return ""


setup(
    name="doctree-rag",
    version="0.1.0",
    description="轻量级文档树索引与推理检索引擎",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="DocTree-RAG Contributors",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[],
    extras_require={
        "pdf": ["PyPDF2>=3.0.0"],
        "tui": ["rich>=13.0.0"],
        "openai": ["openai>=1.0.0"],
        "anthropic": ["anthropic>=0.18.0"],
        "all": [
            "PyPDF2>=3.0.0",
            "rich>=13.0.0",
            "openai>=1.0.0",
            "anthropic>=0.18.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "doctree=doctree_rag.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Text Processing :: Indexing",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
