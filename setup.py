from pathlib import Path
from setuptools import setup, find_packages

HERE = Path(__file__).parent


def read_requirements():
    req_path = HERE / "requirements.txt"
    if not req_path.exists():
        return []
    lines = [l.strip() for l in req_path.read_text().splitlines()]
    # filter out empty lines, local paths or editable markers
    cleaned = [l for l in lines if l and not l.startswith(".") and not l.startswith("#")]
    return cleaned


setup(
    name='ThoraxGuard',
    version='0.1.0',
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=read_requirements(),
    include_package_data=True,
)