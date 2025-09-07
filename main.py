import argparse
import subprocess
import sys


def run(cmd):
    print("+", " ".join(cmd))
    res = subprocess.run(cmd)
    if res.returncode != 0:
        sys.exit(res.returncode)


def main():
    ap = argparse.ArgumentParser(description="Compute positions (with cache) and generate HTML")
    ap.add_argument("--positions", default="data/emoji_positions.json", help="Positions cache path")
    ap.add_argument("--out", default="emoji_map.html", help="Output HTML path")
    ap.add_argument("--force", action="store_true", help="Force recompute of positions")
    args = ap.parse_args()

    pos_args = ["--out", args.positions]
    if args.force:
        pos_args.append("--force")
    run([sys.executable, "compute_positions.py", *pos_args])
    print(f"Positions ready at {args.positions}")
    print(f"Open index.html via a local server to view (see: make serve)")


if __name__ == "__main__":
    main()
