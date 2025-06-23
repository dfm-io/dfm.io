import json
from pathlib import Path


def generate_matrix():
    posts = []
    for p in sorted(Path("posts").glob("*")):
        if not (p / "pyproject.toml").exists():
            continue
        posts.append(p.name)
    s = json.dumps({"post": posts})
    print(f"matrix={s}")


if __name__ == "__main__":
    generate_matrix()
