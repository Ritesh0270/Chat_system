import json

SCRAPED_FILE = "exactink_chat_ready.json"
MANUAL_FILE = "contact_details.json"
OUTPUT_FILE = "final_knowledge.json"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main():
    scraped = load_json(SCRAPED_FILE)
    manual = load_json(MANUAL_FILE)

    if not isinstance(scraped, list):
        raise ValueError("Scraped file must contain a list")
    if not isinstance(manual, list):
        raise ValueError("Manual file must contain a list")

    final_data = manual + scraped
    save_json(OUTPUT_FILE, final_data)

    print(f"[OK] Final merged file saved: {OUTPUT_FILE}")
    print(f"[OK] Total records: {len(final_data)}")


if __name__ == "__main__":
    main()