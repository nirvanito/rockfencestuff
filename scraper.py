from playwright.sync_api import sync_playwright
import csv

#CLOSE EXCEL BEFORE RUNNING
def scrape_atp(max_rank):
    print("browser is running this will take a few seconds ")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        players_data = []
        seen_ranks = set()

        # first range block on the atp site
        range_start = 0
        range_end = 100

        #loop through each 100 player page
        while range_start < max_rank:
            url = (
                f"https://www.atptour.com/en/rankings/singles"
                f"?dateWeek=2025-12-29&rankRange={range_start}-{range_end}"
            )
            print(f"scraping ranks {range_start}-{range_end} now")

            page.goto(url)
            page.wait_for_selector("table", state="attached")

            rows = page.locator("table tbody tr").all()

            for row in rows:
                cols = row.locator("td").all()

                if len(cols) >= 5:
                    rank = cols[0].text_content().strip()
                    raw_name = cols[1].text_content().strip()
                    age = cols[2].text_content().strip()

                    # make sure we dont grab the arrows
                    name = raw_name.split('\n')[-1].strip()

                    # make sure rank is a number and within the rank boundds
                    if rank.isdigit() and int(rank) <= max_rank:
                        players_data.append({
                            "Rank": rank,
                            "Name": name,
                            "Age": age
                        })

            # go to next 100
            range_start = range_end + 1
            range_end = range_start + 99
        # loop break
        csv_filename = f"2025top{max_rank}.csv"
        # >>>>>CLOSE EXCEL BEFORE RUNNING THIS OR IT WONT WORK<<<<<
        with open(csv_filename, mode="w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=["Rank", "Name", "Age"])
            writer.writeheader()
            writer.writerows(players_data)

        print(f"Done scraping {len(players_data)} players to {csv_filename}")
        browser.close()


if __name__ == "__main__":
    scrape_atp(1000)  # change this number to whatever stopping point u want