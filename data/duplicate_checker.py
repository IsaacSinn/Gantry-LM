import argparse
from collections import Counter

def remove_duplicates(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    seen = set()
    deduped = []
    for line in lines:
        stripped = line.rstrip('\n')
        if stripped == "":
            # Add an empty line only if the last line isn't empty
            if not deduped or deduped[-1].rstrip('\n') != "":
                deduped.append(line)
            # ...skip additional consecutive empty lines...
        else:
            if stripped not in seen:
                seen.add(stripped)
                deduped.append(line)
            # ...skip duplicate non-empty lines...
    return deduped

def check_duplicates(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    seen = set()
    duplicates = set()
    for line in lines:
        stripped = line.rstrip('\n')
        if stripped == "":
            # Skip empty lines
            continue
        if stripped in seen:
            duplicates.add(stripped)
        else:
            seen.add(stripped)
    return duplicates

def main():
    parser = argparse.ArgumentParser(description="Remove duplicates from a file (keeping all empty lines) and output a cleaned file")
    parser.add_argument('file', nargs='?', default=r'./data/chatgpt_train', help="Path to the file to process")
    args = parser.parse_args()
    
    new_lines = remove_duplicates(args.file)

    out_file = args.file + ".dedup"
    with open(out_file, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    print(f"Duplicates removed. Clean file written to: {out_file}")

def check():
    parser = argparse.ArgumentParser(description="Check for duplicates in a file")
    parser.add_argument('file', nargs='?', default=r'./data/chatgpt_train', help="Path to the file to process")
    args = parser.parse_args()
    
    with open(args.file, 'r', encoding='utf-8') as f:
         lines = f.read().splitlines()
    counts = Counter(line for line in lines if line.strip() != "")
    duplicates = {line: count for line, count in counts.items() if count > 1}
    if duplicates:
         print("Duplicates and their counts:")
         for line, count in duplicates.items():
             print(f"{line}: {count}")
    else:
         print("No duplicates found")

if __name__ == "__main__":
    # main()
    check()
