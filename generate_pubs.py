#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, unicode_literals

__all__ = []

import ads

# papers = list(ads.query(authors="Foreman-Mackey"))
# print(papers)

journals = {
    "Publications of the Astronomical Society of the Pacific":
        "Pubs. Astr. Soc. Pac.",
    "The Astrophysical Journal": "Astrophys. J.",
    "ArXiv e-prints": "ArXiv",
    "The Astronomical Journal": "Astron. J.",
}


def format_single_author(author):
    try:
        ln, fn = author.split(", ")
    except ValueError:
        return author
    return "{0}, {1}".format(ln, " ".join(f[0] + "." for f in fn.split()))


def parse_authors(article):
    return map(format_single_author, article.author)


def parse_article(article):
    # Get the journal name.
    journal = journals.get(article.pub, None)
    if journal is None:
        print("Unknown journal: {0}".format(article.pub))
        return None

    # Get the ArXiv link for the paper if it exists.
    arxiv_id = None
    for i in article.identifier:
        if i.lower().startswith("arxiv:"):
            arxiv_id = i[6:]
            break

    # Parse the author list.
    authors = parse_authors(article)
    author_list = ", ".join(authors[:-1])
    if len(authors) > 2:
        author_list += ", & " + authors[-1]
    elif len(authors) > 1:
        author_list += " & " + authors[-1]

    # Get the year and title.
    year = article.year
    title = article.title[0]

    # Make a link for the title.
    if arxiv_id is not None:
        title = "[{0}](http://arxiv.org/abs/{1})".format(title, arxiv_id)
    else:
        title = "[{0}](http://adsabs.harvard.edu/abs/{1})" \
            .format(title, article.bibcode)

    # Break out and deal with pre-prints now.
    if journal in ["ArXiv"]:
        assert arxiv_id is not None, "Couldn't parse arXiv ID"
        return False, "1. {0}, {1}, {2}.".format(author_list, year, title)

    # For published papers.
    volume = article.volume
    page = article.page[0]

    return True, "1. {0}, {1}, {2} *{3}* **{4}** 5." \
        .format(author_list, year, title, journal, volume, page)
