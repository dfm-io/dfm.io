---
title: "About"
---

## Hi, I'm Dan Foreman-Mackey

I'm a Research Engineer at Google Deepmind in New York City.
I'm currently working on [JAX](https://github.com/google/jax).
I used to be an astrophysicist who developed some open source scientific software.

### Code

I write a lot of code for work and in my spare time. All my projects live in
[public repositories on GitHub](https://github.com/dfm). Here are some of my most popular research codes:

<script id="code-template" type="x-tmpl-mustache">
{{#codes}}
<li>
    <i><a href="{{homepageUrl}}">{{name}}</a></i> &mdash; {{description}}
</li>
{{/codes}}
{{^codes}}
Unable to load of software.
{{/codes}}
</script>

<ul id="codelist"></ul>

### Publications

My full list of publications is available
[on ADS](http://adsabs.harvard.edu/cgi-bin/nph-abs_connect?return_req=no_params&author=Foreman-Mackey&db_key=PRE)
but here are a few highlights:

<script id="pub-template" type="x-tmpl-mustache">
{{#pubs}}
<li>
    {{authorsFormat}}, {{year}}, <a href="{{url}}"><i>{{title}}</i></a>.
    {{#codeLink}}<br><small>[<a href="{{codeLink}}">code</a>]</small>{{/codeLink}}
</li>
{{/pubs}}
{{^pubs}}
Unable to load publication list.
{{/pubs}}
</script>

<ul id="publist"></ul>

<script src="https://unpkg.com/mustache@latest"></script>
<script>
  var codeMap = {
    "10.1086/670067": "https://github.com/dfm/emcee",
    "10.1088/0004-637X/795/1/64": "https://github.com/dfm/exopop",
    "10.1088/0004-637X/806/2/215": "https://github.com/dfm/ketu",
    "10.21105/joss.00024": "https://github.com/dfm/corner.py",
    "10.3847/0004-6256/152/6/206": "https://github.com/dfm/peerless",
    "10.3847/1538-3881/aa9332": "https://github.com/dfm/celerite",
    "10.3847/2515-5172/aaaf6c": "https://github.com/dfm/celerite-grad",
    "10.21105/joss.01864": "https://github.com/dfm/emcee",
    "10.21105/joss.03285": "https://github.com/exoplanet-dev/exoplanet"
  };

  function formatAuthors(authors) {
    authors = authors.map(author => {
      var tokens = author.split(", ");
      if (tokens.length != 2) return author;
      return tokens[1][0] + ". " + tokens[0];
    });
    if (authors.length == 1) {
      return authors[0];
    } else if (authors.length >= 5) {
      return authors.slice(0, 4).join(", ") + ", et al.";
    }
    return authors.slice(0, authors.length - 1).join(", ") + ", and " + authors[authors.length - 1];
  }

  (() => {
    var codeTemplate = document.getElementById("code-template").innerHTML;
    fetch("https://raw.githubusercontent.com/dfm/cv/main/data/repos.json")
      .then(response => response.json())
      .then(data => {
        data = data.data.user.pinnedItems.edges.map(value => value.node);
        var rendered = Mustache.render(codeTemplate, { codes: data });
        document.getElementById("codelist").innerHTML = rendered;
      })
      .catch(() => {
        var rendered = Mustache.render(codeTemplate, { codes: [] });
        document.getElementById("codelist").innerHTML = rendered;
      });

    var pubTemplate = document.getElementById("pub-template").innerHTML;
    fetch("https://raw.githubusercontent.com/dfm/cv/main/data/pubs.json")
      .then(response => response.json())
      .then(data => {
        // Only first author
        data = data.filter(value => {
          return value.authors[0].startsWith("Foreman-Mackey") && value.doctype == "article";
        });

        // Format authors
        data = data.map(value => {
          value.authorsFormat = formatAuthors(value.authors);
          value.codeLink = codeMap[value.doi];
          value.title = value.title.replace("{\\&}", "&");
          return value;
        });

        var rendered = Mustache.render(pubTemplate, { pubs: data });
        document.getElementById("publist").innerHTML = rendered;
      })
      .catch(() => {
        var rendered = Mustache.render(pubTemplate, { pubs: [] });
        document.getElementById("publist").innerHTML = rendered;
      });
  })();
</script>
