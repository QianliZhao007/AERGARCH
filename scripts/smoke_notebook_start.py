"""Smoke test the first notebook cells in their actual order."""

from pathlib import Path
import json
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    notebook_path = PROJECT_ROOT / "AERGARCH.ipynb"
    nb = json.loads(notebook_path.read_text(encoding="utf-8"))

    env = {"__name__": "__main__"}
    executed = 0
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        exec(source, env)
        executed += 1
        print(f"executed code cell {executed}")
        if executed == 2:
            break

    print("notebook start smoke ok")


if __name__ == "__main__":
    main()
