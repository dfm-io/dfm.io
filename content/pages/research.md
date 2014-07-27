Title: Research
Url: research/
Save_as: research/index.html

I'm a graduate student at NYU working on next generation astronomical data
analysis under the supervision of [David W.
Hogg](http://hoggresearch.blogspot.com/).
My main research interest is the application of probabilistic data analysis
techniques to interesting datasets in astronomy.  These days, I'm mostly
working with [Kepler](http://en.wikipedia.org/wiki/Kepler_(spacecraft)) data;
everything from the raw pixel values to catalog level inferences.  I'm also
interested in the development of [scientific software](/code) and open-source
practices.


## Selected Papers

Below is a list of some recent interesting papers that I've worked on but
[all of my papers are on the
ArXiv](http://arxiv.org/find/all/1/au:foreman_mackey/0/1/0/all/0/1).


* <div class="meta">
    <a class="title" href="http://arxiv.org/abs/1406.3020">Exoplanet
        population inference and the abundance of Earth analogs from
        noisy, incomplete catalogs [1406.3020]</a>
    <span class="authors"><strong>Daniel Foreman-Mackey</strong>,
        David W. Hogg, Timothy D. Morton</span>
  </div>
  <span class="description">In this paper, we develop a framework
    for hierarchical probabilistic inference of exoplanet populations taking
    into account survey completeness, detection efficiency, and observational
    uncertainties.  Applying our method to <a
    href="http://arxiv.org/abs/1311.6806">an existing catalog</a>, we find
    that Earth-like exoplanets are less common than previously thought.  This
    paper comes with <a
    href="http://figshare.com/articles/Exoplanet_population_inference/1051864">publicly
    released data</a> and <a href="https://github.com/dfm/exopop">open-source
    code</a>.</span>

* <div class="meta">
    <a class="title" href="http://arxiv.org/abs/1403.6015">Fast
        Direct Methods for Gaussian Processes and the Analysis
        of NASA Kepler Mission Data [1403.6015]</a>
    <span class="authors">Sivaram Ambikasaran,
        <strong>Daniel Foreman-Mackey</strong>, Leslie Greengard,
        David W. Hogg, Michael O'Neil</span>
  </div>
  <span class="description">My collaborators in applied math have developed
    <a href="http://www.cims.nyu.edu/~sivaram/manuscript/FDSPAPER.pdf">some
    fast algorithms for solving dense linear systems</a>. In this paper, we
    use these algorithms to compute log-determinants and apply these methods
    to model correlated noise in Kepler data.</span>

* <div class="meta">
    <a class="title" href="http://arxiv.org/abs/1202.3665">emcee:
        The MCMC Hammer [1202.3665]</a>
    <span class="authors"><strong>Daniel Foreman-Mackey</strong>, David W.
        Hogg, Dustin Lang, Jonathan Goodman</span>
  </div>
  <span class="description">In this paper, we present a popular open-source
    Markov chain Monte Carlo package written in Python. There is also <a
    href="http://dan.iel.fm/emcee">online documentation</a> and
    MIT-licensed <a href="https://github.com/dfm/emcee">code</a>.</span>


## Software

Most of my job involves writing scientific software. All of it lives on [my
GitHub account](https://github.com/dfm) and some of the highlights are listed
here:

<ul class="projects">

<li>
<a href="http://dan.iel.fm/emcee" class="project">emcee</a> &mdash;
Kick-ass MCMC sampling in Python.
See <a href="http://arxiv.org/abs/1202.3665">the paper</a>.
</li>

<li>
<a href="https://github.com/dfm/george" class="project">George</a> &mdash;
Blazingly fast Gaussian processes for regression.
Implemented in C++ and Python bindings.
See <a href="http://arxiv.org/abs/1202.3665">the paper</a>.
</li>

<li>
<a href="https://github.com/dfm/triangle.py" class="project">triangle.py</a> &mdash;
Simple corner plots (or scatterplot matrices) in matplotlib.
</li>

<li>
<a href="https://github.com/dfm/kplr" class="project">kplr</a> &mdash;
Python bindings to the
<a href="http://archive.stsci.edu/vo/mast_services.html">MAST Kepler API</a>.
</li>

<li>
<a href="http://daft-pgm.org" class="project">Daft</a> &mdash;
Pixel-perfect probabilistic graphical models using matplotlib.
</li>

</ul>
