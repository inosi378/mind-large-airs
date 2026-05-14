import argparse
from pathlib import Path


REQUIRED_FILES = ["news.tsv", "behaviors.tsv"]


def count_lines(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for _ in f:
            count += 1
    return count


def file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def preview_tsv(path: Path, n: int = 3):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            rows.append(line.rstrip("\n").split("\t"))
    return rows


def check_news(news_path: Path):
    print(f"\n[news.tsv] {news_path}")
    print(f"size: {file_size_mb(news_path):.2f} MB")
    n_lines = count_lines(news_path)
    print(f"lines: {n_lines:,}")

    rows = preview_tsv(news_path, 3)
    for idx, row in enumerate(rows, start=1):
        print(f"sample row {idx}: columns={len(row)} news_id={row[0] if row else 'EMPTY'}")
        if len(row) != 8:
            print(f"  WARNING: expected 8 columns, got {len(row)}")

    return n_lines


def check_behaviors(behaviors_path: Path):
    print(f"\n[behaviors.tsv] {behaviors_path}")
    print(f"size: {file_size_mb(behaviors_path):.2f} MB")
    n_lines = count_lines(behaviors_path)
    print(f"lines: {n_lines:,}")

    rows = preview_tsv(behaviors_path, 3)
    for idx, row in enumerate(rows, start=1):
        print(f"sample row {idx}: columns={len(row)} impression_id={row[0] if row else 'EMPTY'}")
        if len(row) != 5:
            print(f"  WARNING: expected 5 columns, got {len(row)}")
            continue

        impression_id, user_id, time, history, impressions = row
        history_count = 0 if not history else len(history.split())
        impression_count = len(impressions.split()) if impressions else 0

        labels = []
        for item in impressions.split():
            if "-" in item:
                labels.append(item.rsplit("-", 1)[-1])

        pos_count = sum(1 for x in labels if x == "1")
        neg_count = sum(1 for x in labels if x == "0")

        print(
            f"  user={user_id}, history={history_count}, "
            f"candidates={impression_count}, positives={pos_count}, negatives={neg_count}"
        )

    return n_lines


def check_dir(data_dir: Path, name: str):
    print("=" * 80)
    print(f"Checking {name}: {data_dir}")

    if not data_dir.exists():
        raise FileNotFoundError(f"{name} directory does not exist: {data_dir}")

    for filename in REQUIRED_FILES:
        path = data_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")

    news_lines = check_news(data_dir / "news.tsv")
    behavior_lines = check_behaviors(data_dir / "behaviors.tsv")

    print(f"\nSummary for {name}")
    print(f"news.tsv lines      : {news_lines:,}")
    print(f"behaviors.tsv lines : {behavior_lines:,}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str, required=True)
    parser.add_argument("--dev_dir", type=str, required=True)
    args = parser.parse_args()

    train_dir = Path(args.train_dir)
    dev_dir = Path(args.dev_dir)

    check_dir(train_dir, "train")
    check_dir(dev_dir, "dev")

    print("\nData check completed.")


if __name__ == "__main__":
    main()