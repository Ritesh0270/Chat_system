import json
import re
from collections import deque
from typing import Iterable, Optional
from urllib.parse import urljoin, urlparse, urldefrag

import requests
from bs4 import BeautifulSoup


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}
SKIP_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp",
    ".pdf", ".zip", ".css", ".js", ".xml", ".ico",
    ".mp4", ".mp3", ".doc", ".docx", ".xls", ".xlsx",
}
CONTENT_TAGS = ["h1", "h2", "h3", "p", "li"]
NOISE_EXACT = {
    "home", "about", "about us", "contact", "privacy policy",
    "learn more", "submit", "read more", "click here", "menu",
    "learn more about us",
}
NOISE_CONTAINS = {
    "curious to know more",
    "fill out the form below",
    "we look forward to speaking with you soon",
    "fostering long-lasting relationships is our goal",
    "we’ll be in touch",
    "[ contact us ]",
    "[ about us ]",
}
SKIP_PATH_KEYWORDS = {
    "/wp-admin", "/wp-login", "/feed", "/tag/", "/author/",
    "/privacy-policy",
}
DROP_SECTION_SLUGS = {
    "docker",
    "general",
    "curious_to_know_more_lets_talk",
}
EXACTINK_SERVICE_MAP = {
    "mobile app": "Custom Mobile Solutions",
    "web app": "Scalable Web Systems",
    "ui/ux": "User-Focused Experience",
    "idea brainstorming": "Creative Concept Development",
    "consulting": "Expert Strategic Guidance",
    "testing": "Quality Assurance Excellence",
    "devops": "Continuous Delivery Automation",
}
PROJECT_KEYWORDS = {
    "bigfin fishing",
    "power play",
    "invito",
    "show buzz",
    "panther city films",
    "medicare comparison",
    "auto insurance quote",
    "health care for pets",
    "island bin cleaners",
    "scrub bin cleaning",
}
DROP_SELECTORS = [
    "header",
    "footer",
    "nav",
    "aside",
    "form",
    ".menu",
    ".navbar",
    ".navigation",
    ".footer",
    ".sidebar",
    ".cookie",
    ".cookies",
    ".popup",
    ".modal",
    ".advertisement",
    ".ads",
]
REQUEST_TIMEOUT = 20


def fetch_html(url: str) -> str:
    response = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    response.encoding = response.apparent_encoding or response.encoding
    return response.text


def normalize_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_key(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"\s+", "_", text)
    return text or "untitled"


def slugify_path(url: str) -> str:
    path = urlparse(url).path.strip("/")
    if not path:
        return "homepage"
    return slugify(path.replace("/", "_"))


def is_same_domain(base_url: str, candidate_url: str) -> bool:
    base_domain = urlparse(base_url).netloc.lower().removeprefix("www.")
    candidate_domain = urlparse(candidate_url).netloc.lower().removeprefix("www.")
    return base_domain == candidate_domain


def should_skip_url(url: str) -> bool:
    parsed = urlparse(url)
    lower = url.lower()

    if lower.startswith(("mailto:", "tel:", "javascript:")):
        return True

    if parsed.scheme not in {"http", "https"}:
        return True

    if any(parsed.path.lower().endswith(ext) for ext in SKIP_EXTENSIONS):
        return True

    if any(keyword in parsed.path.lower() for keyword in SKIP_PATH_KEYWORDS):
        return True

    return False


def is_noise_line(text: str, tag_name: Optional[str]) -> bool:
    lower = text.lower().strip()

    if tag_name not in {"h1", "h2", "h3"} and len(lower) < 15:
        return True
    if lower in NOISE_EXACT:
        return True
    if any(phrase in lower for phrase in NOISE_CONTAINS):
        return True
    if "all rights reserved" in lower:
        return True
    if re.search(r"©\s*\d{4}", lower):
        return True
    if "@" in lower and len(lower) < 40:
        return True
    if lower.count("|") >= 3:
        return True

    return False


def clean_soup(soup: BeautifulSoup) -> BeautifulSoup:
    for tag in soup(["script", "style", "noscript", "svg", "iframe", "aside"]):
        tag.decompose()

    for selector in DROP_SELECTORS:
        for node in soup.select(selector):
            node.decompose()

    return soup


def get_page_title(soup: BeautifulSoup) -> str:
    if soup.title and soup.title.string:
        return normalize_text(soup.title.string)

    h1 = soup.find("h1")
    if h1:
        return normalize_text(h1.get_text(" ", strip=True))

    return "Untitled"


def get_content_root(soup: BeautifulSoup):
    for selector in ["main", "article", "[role='main']", ".entry-content", ".post-content", "body"]:
        node = soup.select_one(selector)
        if node:
            return node
    return soup


def extract_blocks(soup: BeautifulSoup) -> list[dict]:
    blocks = []
    root = get_content_root(soup)

    for tag in root.find_all(CONTENT_TAGS):
        text = normalize_text(tag.get_text(" ", strip=True))
        if not text:
            continue
        if is_noise_line(text, tag.name):
            continue
        blocks.append({"tag": tag.name, "text": text})

    return blocks


def dedupe_blocks(blocks: Iterable[dict]) -> list[dict]:
    seen = set()
    filtered = []

    for block in blocks:
        key = f"{block['tag']}::{normalize_key(block['text'])}"
        if key in seen:
            continue
        seen.add(key)
        filtered.append(block)

    return filtered


def strip_inline_noise(text: str) -> str:
    cleaned = text
    cleaned = re.sub(r"\[\s*contact us\s*\]", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\[\s*about us\s*\]", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"learn\s+more", " ", cleaned, flags=re.IGNORECASE)
    for phrase in NOISE_CONTAINS:
        cleaned = re.sub(re.escape(phrase), " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip(" -|[]")


def split_step_leakage(text: str) -> list[str]:
    if "Step-" not in text:
        return [text]

    parts = re.split(r"(?=Step-\d{2}\s+[A-Za-z&\-\s]+)", text)
    parts = [normalize_text(p) for p in parts if normalize_text(p)]
    return parts or [text]


def classify_category(page: str, heading: str, content: str) -> str:
    key = f"{page} {heading} {content}".lower()
    if "services" in page or "services" in heading.lower():
        return "services"
    if "expertise" in page:
        return "expertise"
    if "about" in page or "journey" in heading.lower() or "growth" in heading.lower():
        return "company_overview"
    if "portfolio" in page or any(project in key for project in PROJECT_KEYWORDS):
        return "projects"
    if "contact" in page:
        return "contact"
    return "general"


def build_sections(blocks: Iterable[dict], source_url: str, page_title: str) -> list[dict]:
    sections = []
    current_heading = None
    current_content = []

    def save_section() -> None:
        nonlocal current_heading, current_content

        if not current_content:
            return

        heading = current_heading or page_title or "general"
        content = normalize_text(" ".join(current_content))
        content = strip_inline_noise(content)
        if len(content) < 30:
            return

        page = slugify_path(source_url)
        category = classify_category(page, heading, content)

        for piece in split_step_leakage(content):
            piece = strip_inline_noise(piece)
            if len(piece) < 30:
                continue
            sections.append(
                {
                    "title": page_title,
                    "page": page,
                    "section": slugify(heading),
                    "heading": heading,
                    "content": piece,
                    "source": source_url,
                    "category": category,
                }
            )

    for block in blocks:
        tag = block["tag"]
        text = block["text"]

        if tag in {"h1", "h2", "h3"}:
            save_section()
            current_heading = text
            current_content = []
        else:
            current_content.append(text)

    save_section()
    return sections


def extract_internal_links(base_url: str, soup: BeautifulSoup) -> set[str]:
    links = set()

    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        full_url = urljoin(base_url, href)
        full_url, _ = urldefrag(full_url)

        if should_skip_url(full_url):
            continue
        if not is_same_domain(base_url, full_url):
            continue

        normalized = full_url.rstrip("/") or full_url
        links.add(normalized)

    return links


def merge_homepage_service_fragments(sections: list[dict]) -> list[dict]:
    homepage_service_bits = []
    kept = []

    for section in sections:
        page = section.get("page", "")
        heading = section.get("heading", "").strip().lower()
        content = section.get("content", "")

        if page == "homepage" and heading in EXACTINK_SERVICE_MAP:
            summary = EXACTINK_SERVICE_MAP[heading]
            text = f"{section['heading']}: {summary}."
            if normalize_key(content) not in normalize_key(text):
                text = f"{text} {content}"
            homepage_service_bits.append(normalize_text(text))
            continue

        kept.append(section)

    if homepage_service_bits:
        kept.append(
            {
                "title": "Exactink",
                "page": "homepage",
                "section": "services_overview",
                "heading": "Services Overview",
                "content": normalize_text(" ".join(homepage_service_bits)),
                "source": "https://www.exactink.com",
                "category": "services",
            }
        )

    return kept


def filter_sections(sections: Iterable[dict]) -> list[dict]:
    filtered = []

    for section in sections:
        heading = section.get("heading", "").strip()
        slug = section.get("section", "").strip().lower()
        content = strip_inline_noise(section.get("content", ""))

        if slug in DROP_SECTION_SLUGS:
            continue
        if normalize_key(heading) in DROP_SECTION_SLUGS:
            continue
        if len(content) < 30:
            continue
        if any(phrase in content.lower() for phrase in NOISE_CONTAINS) and len(content) < 220:
            continue

        section["content"] = content
        filtered.append(section)

    return filtered


def dedupe_sections(sections: Iterable[dict]) -> list[dict]:
    seen_full = set()
    seen_content = set()
    final_sections = []

    for section in sections:
        content_key = normalize_key(section["content"])
        full_key = (
            section["page"],
            section["section"],
            content_key,
        )
        if full_key in seen_full:
            continue

        if content_key in seen_content and len(content_key) < 220:
            continue

        seen_full.add(full_key)
        seen_content.add(content_key)
        final_sections.append(section)

    return final_sections


def prioritize_links(links: set[str]) -> list[str]:
    priority_keywords = ["/about", "/services", "/expertise", "/portfolio", "/contact"]

    def score(link: str) -> tuple[int, str]:
        path = urlparse(link).path.lower()
        for idx, keyword in enumerate(priority_keywords):
            if keyword in path:
                return (idx, path)
        return (len(priority_keywords), path)

    return sorted(links, key=score)


def crawl_website(start_url: str, max_pages: int = 15) -> list[dict]:
    normalized_start = start_url.rstrip("/")
    visited = set()
    queue = deque([normalized_start])
    all_sections = []

    while queue and len(visited) < max_pages:
        url = queue.popleft()
        if url in visited:
            continue

        try:
            html = fetch_html(url)
            soup = BeautifulSoup(html, "html.parser")
            page_title = get_page_title(soup)
            soup = clean_soup(soup)

            blocks = dedupe_blocks(extract_blocks(soup))
            sections = build_sections(blocks, url, page_title)
            sections = filter_sections(sections)
            all_sections.extend(sections)
            visited.add(url)

            print(f"[OK] Crawled: {url} | Title: {page_title} | Sections: {len(sections)}")

            for link in prioritize_links(extract_internal_links(normalized_start, soup)):
                if link not in visited and link not in queue:
                    queue.append(link)

        except Exception as exc:
            print(f"[ERROR] Failed: {url} | {exc}")
            visited.add(url)

    final_sections = merge_homepage_service_fragments(all_sections)
    final_sections = dedupe_sections(final_sections)
    return final_sections


def save_json(data: list[dict], output_file: str = "exactink_chat_ready.json") -> None:
    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)
    print(f"[OK] JSON saved: {output_file}")


if __name__ == "__main__":
    url = "https://www.exactink.com/"
    data = crawl_website(url, max_pages=15)
    save_json(data)
