import hashlib
import json
import sys
from pathlib import Path


def hash_file(sha, file_path: Path) -> str:
    with file_path.open("rb") as f:
        while chunk := f.read(8192):
            sha.update(chunk)


def checksum(path: Path) -> str:
    sha = hashlib.sha256()
    for root, _, files in path.walk():
        if root.is_relative_to(path / ".venv"):
            continue
        for file in files:
            hash_file(sha, root / file)
    return sha.hexdigest()


def generate_matrix():
    posts = ["placeholder"]
    for p in sorted(Path("posts").glob("*")):
        if not (p / "pyproject.toml").exists():
            continue
        ipynb_file = Path("executed") / f"{p.name}.ipynb"
        sha_file = Path("executed") / f"{p.name}.sha.txt"
        if (
            ipynb_file.exists()
            and sha_file.exists()
            and sha_file.read_text().strip() == checksum(p)
        ):
            continue
        posts.append(p.name)
    s = json.dumps({"post": posts})
    print(f"matrix={s}")


def generate_checksums():
    for p in sorted(Path("posts").glob("*")):
        if not (p / "pyproject.toml").exists():
            continue
        sha_file = Path("executed") / f"{p.name}.sha.txt"
        with sha_file.open("w") as f:
            f.write(checksum(p))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "checksums":
        generate_checksums()
    else:
        generate_matrix()
