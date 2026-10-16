# site/

What https://mrswallbreakermaps.com takes from this repository for AI Safety Field Genealogy, and
nothing else.

`site.json` is the whole contract. `title` and `desc` fill the project card on the site's home
page, `preview` is the card's image, and `files` maps each URL under `/genealogy/` on the site to a
file in this repository. The site pins this repository by commit and copies exactly the listed
files at that commit.

The charts are served straight from `viz/`. The landing page is `site/index.html`, not
`viz/index.html`: it links back to the site, groups the charts, and carries a later list of key
findings, caveats and collection debts than `viz/index.html` does.
