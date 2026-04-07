from setuptools import setup

setup(
    name="medtriage-ai",
    version="0.1.0",
    py_modules=[],
    install_requires=[
        "fastapi",
        "uvicorn",
        "openenv-core>=0.2.0"
    ],
    entry_points={
        "console_scripts": [
            "server=server.app:main",
            "medical-triage-serve=server.app:main"
        ]
    },
)
