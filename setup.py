from pathlib import Path
from setuptools import setup, find_packages

HERE = Path(__file__).parent


def read_requirements():
    req_path = HERE / "requirements.txt"
    if not req_path.exists():
        return []
    lines = [l.strip() for l in req_path.read_text().splitlines()]
    # Filter out empty lines, comments, and local/editable-install directives.
    # These are valid in requirements.txt but not in install_requires.
    cleaned = [
        l for l in lines
        if l and not l.startswith((".", "#", "-e", "--editable"))
    ]
    return cleaned


setup(
    name='ThoraxGuard',
    version='0.1.0',
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=read_requirements(),
    include_package_data=True,
)
