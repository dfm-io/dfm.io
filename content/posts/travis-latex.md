---
Title: Continuous integration of academic papers
Date: 2017-07-12
Category: Data Analysis
Slug: travis-latex
Summary: keeping an up-to-date build of your TeX source using GitHub and Travis
Math: true
---

It's becoming more common for astronomers to use continuous integration services like [Travis CI](https://travis-ci.org) to automatically test their code but, as much as I hate to say it, a big part of our job is writing papers.
I am always in search of new procrastination tasks, especially if they can be justified as work, so I was pretty excited to figure out that it is possible to use Travis CI for writing too.
The basic idea is to build the LaTeX source on Travis and force push the PDF to a new branch on GitHub so that there is always a current version of the PDF available online.
Now, before you tell me that I should just be using Authorea, Overleaf, etc., let me say that I am incapable of using a computer without [my heavily customized (neo)vim setup](https://github.com/dfm/dotfiles/blob/master/neovim/init.vim).

I'm sure that other people have done things like this before, but the first time I did it was at [AstroHackWeek 2016](http://astrohackweek.org/2016/).
Since then, I've started using it for [the papers that I'm writing](https://github.com/dfm/celerite/blob/master-pdf/paper/ms.pdf), [lecture materials](https://github.com/dfm/imprs/blob/master-pdf/mcmc/mcmc.pdf), and [my CV](https://github.com/dfm/cv/blob/master-pdf/cv_pubs.pdf) – it comes in surprisingly handy!
[Andy Casey](http://astrowizici.st/) and I have both iterated to come up with a streamlined procedure that doesn't use all of the resources provided by Travis so I wanted to document what I've settled on.

To get started, choose a GitHub repository that has a paper in it.
I'll assume that the paper is in a subdirectory called `paper` and that the TeX file is called `ms.tex`, but it shouldn't be too hard to change these assumptions for your use case.
If your repository isn't already using Travis, you should create a `.travis.yml` file (at the top-level directory of your git repo), log into Travis (using your GitHub account), and enable builds for that repository ([there are resources online to get you started](https://www.google.com/search?q=getting+started+with+travis+ci)).
You'll also need to give Travis push access to your repository.
To do this, [go to your GitHub settings and create a new personal access token](https://github.com/settings/tokens).
Give it a good name and make sure that you enable `repo` access.
Copy this token and go to the settings page for your repository on Travis and add two environment variables:
1. `GITHUB_API_KEY` - this should be set to the personal access token that you created above, and
2. `GITHUB_USER` - this should be set to your username.
While you're in the settings, you might also want to enable the "Build only if .travis.yml is present" option (this will save you from some annoying emails later).

Now that you have Travis set up, here's the minimal `.travis.yml` file that we'll need:

```yaml
sudo: false
language: generic
matrix:
  include:
    - os: linux
      env: TEST_LANG='paper'
script: |
  if [[ $TEST_LANG == paper ]]
  then
    source .ci/build-paper.sh
  fi
```

It should be possible to combine this with any other tests that you're already running.
For [one of my projects](https://github.com/dfm/celerite/blob/master/.travis.yml), I have combined this with testing C++ and several versions of Python and NumPy.

The (yet non-existant) script `build-paper.sh` will check if any changes have been made in the `paper` subdirectory and, if they have, install the [Tectonic](https://tectonic-typesetting.github.io) typesetting package using `conda`, compile the paper from source, and force-push the paper to a new branch called `master-pdf` (assuming you're currently on the `master` branch).
To make this happen, create the file `.ci/build-paper.sh` (at the top-level directory of your git repo; that is, the subdir `.ci` should be at the top level), make `build-paper.sh` executable (`chmod +x .ci/build-paper.sh`), and add the following contents:

```bash
#!/bin/bash -x

if git diff --name-only $TRAVIS_COMMIT_RANGE | grep 'paper/'
then
  # Install tectonic using conda
  wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh;
  bash miniconda.sh -b -p $HOME/miniconda
  export PATH="$HOME/miniconda/bin:$PATH"
  hash -r
  conda config --set always_yes yes --set changeps1 no
  conda update -q conda
  conda info -a
  conda create --yes -n paper
  source activate paper
  conda install -c conda-forge -c pkgw-forge tectonic
  
  # Build the paper using tectonic
  cd paper
  tectonic ms.tex --print
  
  # Force push the paper to GitHub
  cd $TRAVIS_BUILD_DIR
  git checkout --orphan $TRAVIS_BRANCH-pdf
  git rm -rf .
  git add -f paper/ms.pdf
  git -c user.name='travis' -c user.email='travis' commit -m "building the paper"
  git push -q -f https://$GITHUB_USER:$GITHUB_API_KEY@github.com/$TRAVIS_REPO_SLUG $TRAVIS_BRANCH-pdf
fi
```

Now, if you git-add `.ci/build-paper.sh` and push, and if everything went as planned, you should get a new branch called `master-pdf` on GitHub with one file `paper/ms.pdf`.
As the icing on the cake, you can add a badge to your `README` with the image
`https://img.shields.io/badge/PDF-latest-orange.svg?style=flat` pointing to the URL `https://github.com/USERNAME/REPONAME/blob/master-pdf/paper/ms.pdf`
Then you'll get something that looks like this:

[![](https://img.shields.io/badge/PDF-latest-orange.svg?style=flat)](https://github.com/dfm/celerite/blob/master-pdf/paper/ms.pdf)

*(You can click on that to read the most up-to-date version of my most recent paper!)*

If you run into any issues, put them in the comments and let's try to debug.

Thanks to [David W. Hogg for clarifications](https://github.com/dfm/dfm.io/pull/1).

