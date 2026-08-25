"""Split the shared smoke fixture into shards for DataTrove Ray ranks."""

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--shards", type=int, default=4)
    args = parser.parse_args()

    lines = [line for line in Path(args.source).read_text(encoding="utf-8").splitlines() if line.strip()]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for index in range(args.shards):
        shard_lines = lines[index :: args.shards]
        (output / f"part-{index:05d}.jsonl").write_text("\n".join(shard_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
