#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Ref: https://www.datisticsblog.com/2018/08/post_with_jupyter/

import re
from pathlib import Path

import nbformat
import yaml
from nbconvert import MarkdownExporter
from nbconvert.preprocessors import Preprocessor
from traitlets.config import Config

inline_pat = re.compile(r"\$(.+?)\$", flags=re.M | re.S)
block_pat = re.compile(r"\$\$(.+?)\$\$", flags=re.M | re.S)


class CustomPreprocessor(Preprocessor):
    def preprocess(self, nb, resources):
        for index, cell in enumerate(nb.cells):
            if cell.cell_type == "code" and not cell.source:
                nb.cells.pop(index)
            elif cell.cell_type == "code" and cell.source.startswith("%matplotlib"):
                nb.cells.pop(index)
            else:
                nb.cells[index], resources = self.preprocess_cell(
                    cell, resources, index
                )
        return nb, resources

    def preprocess_cell(self, cell, resources, cell_index):
        if cell.cell_type == "code":
            cell.source = cell.source.strip()
        else:
            text = "$$".join(
                inline_pat.sub(r"`$\1$`", t) for t in cell.source.split("$$")
            )
            cell.source = block_pat.sub(r"<div>$$\1$$</div>", text)
        return cell, resources


def render_notebook(path, metadata):
    meta = yaml.safe_load(metadata)
    slug = meta["slug"]

    with path.open() as fp:
        notebook = nbformat.read(fp, as_version=4)

    c = Config()
    c.MarkdownExporter.preprocessors = [CustomPreprocessor]
    markdown_exporter = MarkdownExporter(config=c)
    markdown, resources = markdown_exporter.from_notebook_node(notebook)
    markdown = f"---\n{metadata.strip()}\n---\n\n" + markdown

    output = Path(f"./content/posts/notebook.{slug}.md")
    with output.open("w") as f:
        f.write(markdown)

    if "outputs" in resources.keys():
        out_path = Path(f"./static/posts/{slug}")
        if not out_path.exists():
            out_path.mkdir(parents=True)
        for key in resources["outputs"].keys():
            with (out_path / key).open("wb") as f:
                f.write(resources["outputs"][key])


def render_notebooks():
    for p in sorted(Path("posts").glob("*")):
        executed = Path("executed") / f"{p.name}.ipynb"
        if not executed.exists():
            continue
        with (p / "metadata.yml").open("r") as f:
            metadata = f.read()
        render_notebook(executed, metadata)


if __name__ == "__main__":
    render_notebooks()
