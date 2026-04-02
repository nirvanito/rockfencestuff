from playwright.sync_api import sync_playwright
import csv
import time
import random

#CLOSE EXCEL BEFORE RUNNING
def scrape_atp(max_rank):
    print("browser is running this will take a few seconds ")

    # stuff to scrape (i excluded 2020 and 2021 bc covid)
    dates = [
        ("2025", "2025-12-29"),
        ("2024", "2024-12-30"),
        ("2023", "2023-12-25"),
        ("2022", "2022-12-26"),
        ("2019", "2019-12-30"),
        ("2018", "2018-12-31"),
        ("2017", "2017-12-25"),
        ("2016", "2016-12-26"),
        ("2015", "2015-12-28"),
        ("2014", "2014-12-29"),
        ("2013", "2013-12-30"),
        ("2012", "2012-12-31"),
        ("2011", "2011-12-26"),
        ("2010", "2010-12-27"),
        ("2009", "2009-12-28"),
        ("2008", "2008-12-29"),
        ("2007", "2007-12-31"),
        ("2006", "2006-12-25"),
        ("2005", "2005-12-26"),
        ("2004", "2004-12-27"),
        ("2003", "2003-12-29"),
        ("2002", "2002-12-30"),
        ("2001", "2001-12-31"),
        ("2000", "2000-12-25"),
    ]

    all_players_data = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        for year, date_str in dates:
            print(f"\n=== Scraping year {year} (date: {date_str}) ===")

            # first range block on the atp site
            range_start = 0
            range_end = 100

            #loop through each 100 player page
            while range_start < max_rank:
                url = (
                    f"https://www.atptour.com/en/rankings/singles"
                    f"?dateWeek={date_str}&rankRange={range_start}-{range_end}"
                )
                print(f"  scraping ranks {range_start}-{range_end} now")

                # stuff to stop being rate limited
                success = False
                for attempt in range(1, 4):
                    try:
                        page.goto(url, wait_until="domcontentloaded", timeout=45000)
                        page.wait_for_selector("table", state="attached", timeout=20000)
                        success = True
                        break
                    except Exception as e:
                        wait_sec = attempt * 5 + random.uniform(1, 3)
                        print(f"  Attempt {attempt} failed for {year} ranks {range_start}-{range_end}: {type(e).__name__}. Retrying in {wait_sec:.1f}s...")
                        time.sleep(wait_sec)

                if not success:
                    print(f"  SKIPPING {year} ranks {range_start}-{range_end} after 3 failed attempts.")
                    range_start = range_end + 1
                    range_end = range_start + 99
                    continue

                # stop being api rate limited
                time.sleep(random.uniform(1.5, 3.5))

                rows = page.locator("table tbody tr").all()

                if not rows:
                    print(f"  WARNING: Table found but no rows for {year} ranks {range_start}-{range_end} — skipping.")
                    range_start = range_end + 1
                    range_end = range_start + 99
                    continue

                for row in rows:
                    cols = row.locator("td").all()

                    if len(cols) >= 5:
                        rank = cols[0].text_content().strip()
                        raw_name = cols[1].text_content().strip()
                        age = cols[2].text_content().strip()

                        # make sure we dont grab the arrows
                        name = raw_name.split('\n')[-1].strip()

                        # make sure rank is a number and within the rank bounds
                        if rank.isdigit() and int(rank) <= max_rank:
                            all_players_data.append({
                                "Year": year,
                                "Rank": rank,
                                "Name": name,
                                "Age": age
                            })

                # go to next 100
                range_start = range_end + 1
                range_end = range_start + 99

        browser.close()

    # >>>>>CLOSE EXCEL BEFORE RUNNING THIS OR IT WONT WORK<<<<<
    csv_filename = f"2000-2025top{max_rank}.csv"
    with open(csv_filename, mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["Year", "Rank", "Name", "Age"])
        writer.writeheader()
        writer.writerows(all_players_data)

    print(f"\nDone! Scraped {len(all_players_data)} total player-rows across {len(dates)} years.")
    print(f"Saved to {csv_filename}")


if __name__ == "__main__":
    scrape_atp(1000)  # change this number to whatever stopping point u want