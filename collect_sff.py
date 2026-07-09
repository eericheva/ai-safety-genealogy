"""Harvest SFF (Survival and Flourishing Fund) annual grant totals from the fund's own
home page, which publishes a year -> $ summary ("SFF has organized ~$152MM ...") plus
the latest round announcements. We write the authoritative per-year totals to
`data/sff_annual_totals.csv`, each row carrying the source URL.

Why this exists: SFF's per-year totals are the cleanest machine-readable statement of how
much SFF moved each year, straight from the funder. The project's events.csv holds
itemised SFF `grant_out` rows (main + flexHEGs etc.); this reference file lets the post
cite the funder's own headline totals and cross-check the itemised figures.

The values on the page are funder-rounded ($MM). The precise latest-round figure
($34.92MM for SFF-2025) is also extracted from the announcement text when present.

No fabricated numbers, fails loudly (no try/except). Run inside tmux with a watchdog:
  tmux new-session -d -s sff 'python3 collect_sff.py 2>&1 | tee logs/sff.log'
"""

import re
import urllib.request
import pandas as pd

OUT_CSV = "data/sff_annual_totals.csv"
PAGE_URL = "https://survivalandflourishing.fund/"
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
      "(KHTML, like Gecko) Version/16.0 Safari/605.1.15")


def fetch_text(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    html = urllib.request.urlopen(req, timeout=60).read().decode(errors="replace")
    text = re.sub(r"<[^>]+>", " ", html)
    return re.sub(r"\s+", " ", text)


def main():
    print("START collect_sff")
    text = fetch_text(PAGE_URL)

    pairs = re.findall(r"(20\d\d)\s*\$\s*([\d.]+)\s*MM", text)
    assert pairs, "no 'YEAR $NMM' annual totals found on SFF page (structure changed)"

    precise = re.search(r"distributed in association with this round is \$\s*([\d.]+)\s*MM", text)
    precise_2025 = float(precise.group(1)) * 1e6 if precise else None

    rows = []
    for year, mm in pairs:
        year = int(year)
        amount = round(float(mm) * 1e6)
        note = "funder-rounded annual total ($MM) from SFF home page"
        if year == 2025 and precise_2025 is not None:
            amount = round(precise_2025)
            note = "precise SFF-2025 round total from announcement text"
        rows.append({"fund": "SFF", "year": year, "amount_usd": amount, "note": note,
                     "source": PAGE_URL})

    out = pd.DataFrame(rows).drop_duplicates("year").sort_values("year").reset_index(drop=True)
    out.to_csv(OUT_CSV, index=False)
    for _, x in out.iterrows():
        print(f"rc=0 SFF {x['year']} ${x['amount_usd']:,}")
    print(f"wrote {OUT_CSV}: {len(out)} years, {out.year.min()}-{out.year.max()}, "
          f"cumulative ${out.amount_usd.sum():,}")
    print("ALL DONE")


if __name__ == "__main__":
    main()
