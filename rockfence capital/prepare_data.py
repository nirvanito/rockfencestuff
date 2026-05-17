import pandas as pd
import unicodedata
import os
import difflib

def strip_accents(text):
    if not isinstance(text, str):
        return text
    return ''.join(c for c in unicodedata.normalize('NFKD', text) if unicodedata.category(c) != 'Mn')

def normalize_name(name):
    # lower case, strip accents, replace hyphens with space
    if not isinstance(name, str):
        return ""
    name = strip_accents(name).lower()
    name = name.replace('-', ' ').replace(',', '')
    # split into words and sort them to handle "Last First" vs "First Last"
    words = name.split()
    words.sort()
    return " ".join(words)

def main():
    years = [2015, 2016, 2017, 2018, 2019, 2022, 2023, 2024, 2025]
    
    # Load top 1000
    top1000_path = "2000-2025top1000rankedv2.csv"
    df_top = pd.read_csv(top1000_path)
    
    # Filter years
    df_top = df_top[df_top['Year'].isin(years)]
    
    out_path = "filtered_top_players_earnings.xlsx"
    with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
        for year in years:
            print(f"Processing {year}...")
            df_top_year = df_top[df_top['Year'] == year].copy()
            
            cleaned_path = f"cleaned earnings/{year}cleaned.csv"
            if not os.path.exists(cleaned_path):
                print(f"Warning: {cleaned_path} not found.")
                continue
                
            df_clean = pd.read_csv(cleaned_path)
            
            # normalize names in clean
            df_clean['norm_name'] = df_clean['Player'].apply(normalize_name)
            
            clean_names = df_clean['norm_name'].tolist()
            
            matched_count = 0
            out_rows = []
            
            for idx, row in df_top_year.iterrows():
                orig_name = row['Name']
                age = row['Age']
                rank = int(row['Rank'])
                norm = normalize_name(orig_name)
                
                # Match
                match = None
                # exact match on norm
                exact_matches = df_clean[df_clean['norm_name'] == norm]
                if not exact_matches.empty:
                    match = exact_matches.iloc[0]
                else:
                    # fuzzy match
                    close = difflib.get_close_matches(norm, clean_names, n=1, cutoff=0.85)
                    if close:
                        match = df_clean[df_clean['norm_name'] == close[0]].iloc[0]
                
                if match is not None:
                    matched_count += 1
                    out_rows.append({
                        'rank': rank,
                        'player name': orig_name,
                        'age': age,
                        'YTD': float(match['YTD']),
                        'singles': float(match['Singles']),
                        'doubles': float(match['Doubles'])
                    })
            print(f"  Matched {matched_count} players out of {len(df_top_year)}")
                    
            df_out = pd.DataFrame(out_rows)
            df_out = df_out.sort_values(by='rank', ascending=True)
            
            df_out.to_excel(writer, sheet_name=str(year), index=False)
            
    print(f"Done. Saved to {out_path}")

if __name__ == "__main__":
    main()
