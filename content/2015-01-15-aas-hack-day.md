Title: The Hack Day at AAS 225
Date: 2015-01-15 0:00
Category: Hacking
Slug: aas-hack-day
Summary: The wrap-up post from the hack day at the 225th AAS meeting.
Image: /images/hack-day.jpg
Position: bottom
Status: draft

At the last minute, my friend [Adrian P-W](http://adrian.pw/) and I found
ourselves in charge of chairing the Hack Day at the 225th meeting of the
American Astronomical Society that happened Jan 5–8 in Seattle.
This is a job usually left to my charismatic PhD advisor,
[Hogg](http://cosmo.nyu.edu/hogg/), who wrote blog posts about the previous
AAS hack days
([2013](http://hoggresearch.blogspot.com/2013/01/aas-hack-day_10.html) and
[2014](http://hoggresearch.blogspot.com/2014/01/aas-223-day-4-aas-hack-day.html);
you can also find [summary posts on
Astrobetter](http://www.astrobetter.com/tag/hackday/)).
Hack days are a relatively new addition to this meeting and to the
astronomical community but they are always fun and productive.
This day was no exception!

There were about 75 people in the room and roughly half had previously
attended a hack day of some sort.
The day started with hack pitches then after a quick coffee break, we got down
to work.
The main point of this hack day is to spend a full day focusing on a single
project that you don't normally have time for but that you've been wanting to
work on or learn about.
The hacks are by no means required to be code-related but for practical and
sociological reasons, they often are.

In the afternoon, after about five hours of hacking, we regrouped and everyone
reported on what they had been up to.
I don't have space to list all the awesome hacks but here are a few examples:

1. Angus (Oxford/CfA) and Montet (Caltech) built a Twitter "troll-bot" that
   listened for people tweeting about air travel (optionally with the
   #academicnomad hash tag) and responded with an estimate of their carbon
   footprint. Unfortunately it looks like they have since been banned from
   Twitter for violating the terms of service.
   [[code](https://github.com/RuthAngus/trollbot)]

2. Davenport (UW), Morris (UW), Fiore-Silvast (UW), DeStefano (UW),
   Holachek (ADS), and Bianco (NYU) extended [Davenport et
   al. (2013)](http://arxiv.org/abs/1403.3091)'s study of the gender
   distribution of question askers at the meeting. One key result from this
   year's analysis was that if the *first* question is asked by a woman, the
   gender distribution of subsequent questions is more representative of the
   meeting's overall gender distribution (60% M / 40% F) versus the average
   distribution of question askers over all sessions (80% M / 20% F).
   What causes of this difference?
   [[code](https://github.com/jradavenport/aas225-gender),
   [data](https://github.com/jradavenport/aas225-gender/blob/master/data.csv),
   and
   [analysis](http://nbviewer.ipython.org/github/jradavenport/aas225-gender/blob/master/analysis.ipynb)]

3. Partly to answer some of the questions raised by results like the above
   gender study in more detail, Schwamb (ASIAA), Salyk (NOAO), and Avestruz
   (Yale) developed [a
   survey](https://docs.google.com/forms/d/1mPxiaTIKUBl2BAt1KF6OWJ2LOdsQCdfMk-KTbn9vMgc/viewform?c=0&w=1)
   aimed at answering the question: "why do people ask questions?" Some
   results were in but a full analysis wasn't finished by the end of the hack
   day. I'm looking forward to hearing what they find (and I think
   they're still accepting responses)!
   [[survey](https://docs.google.com/forms/d/1mPxiaTIKUBl2BAt1KF6OWJ2LOdsQCdfMk-KTbn9vMgc/viewform?c=0&w=1)]

4. Inspired by the visualizations created for [this
   article](http://nautil.us/issue/19/illusions/a-quick-spin-around-the-big-dipper)
   (that she wrote), Ash (Columbia) figured out how to use Worldwide Telescope
   to render the constellations in 3D and view them from different angles.
   It turns out that constellations are just projection effects ;-).

5. For all of your music discovery needs, DJ Carly Sagan (NYC), Rice
   (CUNY/AMNH), and Lee (Harvard) started a collaborative playlist on
   Spotify for astronomy related tunes. It currently has 637 songs (two days
   of constant listening and Total Eclipse of the Heart is only on there about
   six times) from many different contributors. Add your favorite astrosongs
   now!
   [[playlist](https://play.spotify.com/user/djcarlysagan/playlist/5BLvisuoHWxYaoXaLBN2WD), and
    [instructions](http://www.astrobetter.com/wiki/tiki-index.php?page=Astronomy+Songs)]

6. To study gendered trends in publication practices and coauthorship, Bianco
   (NYU), Morehead (PSU), Wang (NYU) Li (Swarthmore), Senchyna (UW), and Lee
   (Harvard) downloaded about 5000 papers from ADS and estimated the genders
   of the first few authors using first names. Most of the hack day was spent
   scraping and managing the
   data (that's called "data science") so they didn't have time to come to any
   solid conclusions but they did produce an awesome dataset.
   [[code](https://github.com/fedhere/ADSgenderclustering)]

9. Rogers & Donaldson (MAST) collaborated to put together a prototype of a
   very lightweight visual interface to the MAST catalogs and imaging. The app
   is web-based but it is designed to used on mobile devices.
   I think that this will be useful for both research and teaching.

7. VanderPlas (UW), Douglas (Columbia), Morton (Princeton), Peters (Drexel),
   and Hollowood (UCSC) worked together to develop a consistent API for time
   series analysis to be incorporated into the [AstroML
   package](https://github.com/astroML/astroML). By the end of the hack day,
   they had implemented Lomb-Scargle and Supersmoother complete with unit
   tests and a full API specification. That's how I like to see code
   released!
   [[code](https://github.com/astroML/periodogram)]

8. Finkbeiner (CfA) hacked the IDL image-viewing platform, ATV, to show a
   movie of posterior samples instead of static figures. Everyone in the room
   gasped when he demonstrated that you can zoom, pan, *etc.* as you normally
   would while the samples continue to animate. How do we include this in
   papers now? Finkbeiner uses IDL and the classic email-me-to-get-the-code
   license but this is impressive enough work that I guess we'll have to
   forgive him.
   [[code access](mailto:Douglas.Finkbeiner@gmail.com)]

A huge thanks goes to Kelle Cruz (CUNY/AMNH) and Meg Schwamb (ASIAA) for
organizing the event, Debbie Kovalsky (AAS Meetings) for logistics, and our
sponsors, Northrop Grumman and LSST, for the snacks and lunch.
I'd also like to thank everyone who attended for making it such a fun day.

Please consider joining us for Hack Day at AAS 227 in [sunny
Florida](http://www.marriott.com/hotel-info/mcogp-gaylord-palms-resort-and-convention-center/gaylord-palms-entertainment/gwzd4ug/seasonal.mi)!
Programming expertise is *not* necessary and hacks do not even need to be code
based.
One example might be [to make more knitted astronomy
mascots](https://twitter.com/ashpags/status/552709347944312833) at the Hack
Days.
If you have any questions about hack day, want to participate but have
hesitations, please feel free to reach out to
[me](https://tiwtter.com/exoplaneteer), [Kelle](https://twitter.com/kellecruz)
or any of the other organizers.
