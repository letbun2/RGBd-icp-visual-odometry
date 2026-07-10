import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rgbd_vo.dataset import write_associations


def main() -> None:
    parser = argparse.ArgumentParser(description="Create TUM RGB-depth association file.")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-difference", type=float, default=0.02)
    args = parser.parse_args()

    output = args.output or (args.dataset / "associations.txt")
    count = write_associations(args.dataset, output, max_difference=args.max_difference)
    print(f"Wrote {count} associations to {output}")


if __name__ == "__main__":
    main()
