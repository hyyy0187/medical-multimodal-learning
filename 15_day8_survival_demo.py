"""Compatibility entry point; the full synthetic tutorial is in tutorials/."""

from pathlib import Path
import runpy


def main():
    tutorial = Path(__file__).resolve().parent / "tutorials" / "15_day8_survival_demo.py"
    return runpy.run_path(str(tutorial), run_name="__main__")


if __name__ == "__main__":
    main()
