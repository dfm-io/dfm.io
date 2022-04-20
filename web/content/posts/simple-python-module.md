---
Title: What if I want to reuse my Python functions?
Date: 2021-03-18
Slug: simple-python-module
---

This post is an introduction to packaging Python code aimed at scientists
(although the advice is probably more general) who want to be able to reuse
Python functions that they have written as part of a common Jupyter
notebook-based development workflow. It is not meant as a complete discussion of
Python packaging for scientific software. Some day I'd love to write more about
that (because I have so many thoughts and opinions), but in the meantime I'll
direct you to other resources like the excellent [OpenAstronomy Python packaging
guide](https://packaging-guide.openastronomy.org/en/latest/), if you want more
details. Instead, in this post I'll focus on how you can support the most
important user of your code: you! With this in mind, the procedure described
here is meant as a quick-and-dirty first step and **it's definitely not a
description of the best practices**, but it might be enough for many
researchers.

I decided to write this post because I didn't know of a good link to share with
collaborators who were at the (all too common) point in their development cycle
where they are happy with some of the functions that they've written in a
Jupyter notebook and find themselves copying and pasting that definition between
notebooks. There's a lot to say about a Jupyter-based research workflow like
this, but I think you'll know that point where it's hard to tell what is
scratch/exploratory work and what is "production" code. I can't answer that
question, but I can provide some tips for moving code to an importable module.

## Required files

As an example, let's imagine that we want to move a function for loading data to
a module. This might be a good place to start because a function like this
probably doesn't change very often. Our goal here is to get something like

```python
import cool_science

data = cool_science.data.load_with_numpy("/path/to/data")
```

instead of that 150 line cell that you've copied into 12 different notebooks
called `Untitled.ipynb` (harsh, but you know it's true!).

To do this, we'll create 3 files with the following directory layout:

```
├── setup.py
└── src
    └── cool_science
        └── __init__.py
        └── data.py
```

## Where to put the actual code

When discussing these files, let's start at the bottom with `data.py` in the
`src/cool_science` subdirectory. This is where we're going to put the code for
our function, moved from our Jupyter notebook:

```python
# File: src/cool_science/data.py
__all__ = ["load_with_numpy"]

import numpy as np

def load_with_numpy(filename):
    # ...
```

This file is called `data.py` and this becomes a "submodule" called `data` of
our `cool_science` package. By comparison, you could also create a file called
`plotting.py`, for example, with functions to making plots that you would access
using the `cool_science.plotting` module. You don't have to structure your code
this way (all your functions could live in the top level module, for example),
but I often find it useful to structure things this way.

## Boilerplate

Now that we've moved our function to a file, we need some boilerplate code for
making this code installable and importable. First, in the `__init__.py` file,
we're going to list our submodules:

```python
# File: src/cool_science/__init__.py
__all__ = ["data"]

from . import data
```

This is not absolutely necessary, it would be fine for this file to be empty
but, in that case, you would need to import the submodule directly:

```python
import cool_science.data  # instead of `import cool_science`
```

Finally, the `setup.py` file tells Python how to install this code. There are a
lot of options that you can set in this file (and in bigger projects, you might
actually use a different file such as `pyproject.toml` or `setup.cfg`), but
we're going to keep things very simple here and just do the bare minimum:

```python
# File: setup.py
from setuptools import find_packages, setup

setup(
    name="cool_science",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)
```

## Usage

Now that we have our module set up, we can install it as follows:

```bash
python -m pip install -e .
```

where the `-e` flag stands for "editable", which means that you can change the
`data.py` file and use those changes without re-installing.

After installation, you should be able to execute:

```python
import cool_science
```

in a Jupyter notebook or Python script, and use your fancy science functions. If
this doesn't work, you might need to restart your Jupyter kernel or make sure
that you're using the same Python to run your code as you used to install above.

I mentioned above that the `-e` flag lets you make changes to your code and use
them without reinstalling. This is true, but if you're working in a Jupyter
notebook, you will need to restart your kernel after making changes to your
module (there are `%reload` magic functions, but I've never had much success
getting these to work).
