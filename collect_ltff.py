"""Harvest LTFF (EA Funds Long-Term Future Fund) payout reports straight from the
funds.effectivealtruism.org page, which ships the full report list as machine-readable
JSON in its Next.js `__NEXT_DATA__` blob (Contentful-backed). We pull every payout
report's title, date, total amount and grantee count and write them to
`data/ltff_payouts.csv`.

Why this exists: the project's LTFF money spine came from vipulnaik, which stops ~2023.
This collector reads the primary funder page directly, so LTFF coverage extends to its
latest published report (currently the 'May 2023 to March 2024' round). Each row carries
the source URL (the funder page + the report's own EA-Forum permalink) so nothing is
'from the head' and everything is reproducible.

Methodology note: EA-Funds report totals are ALL LTFF grants (all longtermist causes),
NOT the AI-safety-only slice — so this file is a documented reference source, kept
separate from the AI-safety-filtered LTFF rows in events.csv (do not sum the two).

No fabricated numbers, fails loudly (no try/except). Run inside tmux with a watchdog:
  tmux new-session -d -s ltff 'python3 collect_ltff.py 2>&1 | tee logs/ltff.log'
"""

import json
import re
import urllib.request
import pandas as pd

OUT_CSV = "data/ltff_payouts.csv"
PAGE_URL = "https://funds.effectivealtruism.org/funds/far-future"
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
      "(KHTML, like Gecko) Version/16.0 Safari/605.1.15")


def fetch_next_data(url):
    """Return the parsed __NEXT_DATA__ JSON object embedded in the EA-Funds page."""
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    html = urllib.request.urlopen(req, timeout=60).read().decode()
    m = re.search(r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>', html, re.S)
    assert m, "no __NEXT_DATA__ blob on EA-Funds page (page structure changed)"
    return json.loads(m.group(1))


def main():
    print("START collect_ltff")
    data = fetch_next_data(PAGE_URL)
    reports = data["props"]["pageProps"]["payoutReports"]
    assert reports, "payoutReports empty (Contentful shape changed)"

    rows = []
    for r in reports:
        f = r["fields"]
        date = f["date"]                              # ISO date of the report
        year = int(date[:4])
        rows.append({
            "fund": "LTFF",
            "report_title": f["title"],
            "date": date,
            "year": year,
            "amount_usd": f["amount"],
            "n_grantees": f.get("numberOfGrantees", ""),
            "last_included_donation_date": f.get("lastIncludedDonationDate", ""),
            "source": f.get("eaForumLink") or PAGE_URL,
        })

    out = pd.DataFrame(rows).sort_values("date", ascending=False).reset_index(drop=True)
    out.to_csv(OUT_CSV, index=False)
    for _, x in out.iterrows():
        print(f"rc=0 {x['date']} {x['report_title'][:48]!r} ${x['amount_usd']:,}")
    print(f"wrote {OUT_CSV}: {len(out)} reports, {out.year.min()}-{out.year.max()}, "
          f"total ${out.amount_usd.sum():,}")
    print("ALL DONE")


if __name__ == "__main__":
    main()
