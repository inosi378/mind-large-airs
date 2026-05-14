import argparse
import shutil
from pathlib import Path


def copy_first_n_lines(src: Path, dst: Path, n: int) -> int:
    copied = 0
    with src.open("r", encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for i, line in enumerate(fin):
            if i >= n:
                break
            fout.write(line)
            copied += 1
    return copied


def validate_mind_dir(data_dir: Path, name: str):
    news_path = data_dir / "news.tsv"
    behaviors_path = data_dir / "behaviors.tsv"

    if not data_dir.exists():
        raise FileNotFoundError(f"{name} directory does not exist: {data_dir}")
    if not news_path.exists():
        raise FileNotFoundError(f"{name} news.tsv not found: {news_path}")
    if not behaviors_path.exists():
        raise FileNotFoundError(f"{name} behaviors.tsv not found: {behaviors_path}")

    return news_path, behaviors_path


def make_subset(src_dir: Path, out_dir: Path, n_behaviors: int, name: str):
    news_src, behaviors_src = validate_mind_dir(src_dir, name)

    out_dir.mkdir(parents=True, exist_ok=True)

    news_dst = out_dir / "news.tsv"
    behaviors_dst = out_dir / "behaviors.tsv"

    print(f"\n[{name}]")
    print(f"source dir : {src_dir}")
    print(f"output dir : {out_dir}")

    print(f"copy news.tsv -> {news_dst}")
    shutil.copy2(news_src, news_dst)

    print(f"copy first {n_behaviors:,} lines of behaviors.tsv -> {behaviors_dst}")
    copied = copy_first_n_lines(behaviors_src, behaviors_dst, n_behaviors)

    print(f"copied behaviors: {copied:,}")

    if copied < n_behaviors:
        print(
            f"WARNING: requested {n_behaviors:,} behavior lines, "
            f"but source only had {copied:,} lines."
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str, required=True)
    parser.add_argument("--dev_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--train_n", type=int, required=True)
    parser.add_argument("--dev_n", type=int, required=True)
    args = parser.parse_args()

    train_dir = Path(args.train_dir)
    dev_dir = Path(args.dev_dir)
    out_dir = Path(args.out_dir)

    make_subset(
        src_dir=train_dir,
        out_dir=out_dir / "train",
        n_behaviors=args.train_n,
        name="train",
    )

    make_subset(
        src_dir=dev_dir,
        out_dir=out_dir / "dev",
        n_behaviors=args.dev_n,
        name="dev",
    )

    print("\nSubset creation completed.")


if __name__ == "__main__":
    main()