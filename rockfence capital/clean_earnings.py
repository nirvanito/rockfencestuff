import glob
import re
import os
import csv

def parse_header(line):
    cols = ['Career', 'YTD', 'Singles', 'Doubles']
    indices = [(col, line.upper().find(col.upper())) for col in cols]
    indices = [x for x in indices if x[1] != -1]
    indices.sort(key=lambda x: x[1])
    return [x[0] for x in indices]

def clean_money(val):
    return val.replace('$', '').replace(',', '').strip()

def clean_file(filepath, out_dir):
    filename = os.path.basename(filepath)
    out_filepath = os.path.join(out_dir, filename.replace('raw', 'cleaned'))
    
    print(f'Processing {filename} -> {os.path.basename(out_filepath)}')
    
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
        
    order = []
    for line in lines:
        if 'Rank' in line and 'Player' in line and ('YTD' in line or 'Career' in line):
            order = parse_header(line)
            break
            
    if not order:
        print(f'Warning: Could not find header in {filename}, skipping.')
        return
        
    out_header = ['Rank', 'Player', 'YTD', 'Singles', 'Doubles', 'Career']
    
    cleaned_rows = []
    for i, line in enumerate(lines):
        orig_line = line
        line = line.strip().replace('\u00a0', ' ').replace('"', '')
        
        # In 2019 there is a line with just 'Money'. We skip it if the line stripped equals 'Money'
        if not line or 'Prize Money' in line or 'Rank' in line or 'Page' in line or line == 'Money':
            continue
            
        m_rank = re.search(r'^(\d+T?)', line)
        if not m_rank:
            continue
            
        rank = m_rank.group(1)
        
        amounts = re.findall(r'\$[\d,]+|\b0\b', line)
        
        if len(amounts) >= 4:
            amounts = amounts[:4]
            
            first_amount_idx = line.find(amounts[0])
            name_raw = line[m_rank.end():first_amount_idx]
            player = name_raw.strip(' ,\'\t')
            
            if player.endswith(','):
                player = player[:-1].strip()
                
            vals = {}
            for col_name, amt in zip(order, amounts):
                vals[col_name] = clean_money(amt)
                
            row = {
                'Rank': rank,
                'Player': player,
                'YTD': vals.get('YTD', '0'),
                'Singles': vals.get('Singles', '0'),
                'Doubles': vals.get('Doubles', '0'),
                'Career': vals.get('Career', '0')
            }
            cleaned_rows.append(row)
            
    if cleaned_rows:
        with open(out_filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=out_header)
            writer.writeheader()
            writer.writerows(cleaned_rows)
        print(f'  Saved {len(cleaned_rows)} rows.')
    else:
        print(f'  No data found to save.')

def main():
    os.makedirs('cleaned earnings', exist_ok=True)
    for f in sorted(glob.glob('raw earnings/*.csv')):
        clean_file(f, 'cleaned earnings')

if __name__ == '__main__':
    main()
