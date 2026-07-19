"""
Scrape every article on https://helpegypt.noon.com/portal/en/kb and save each
one as a PDF.

Requires Playwright (a real headless browser), since the KB is a JS-rendered
Zoho Desk portal and the "Download as PDF" button is a JS action, not a
static link.

Setup (run once):
    pip install playwright
    playwright install chromium

Run:
    python scrape_kb_to_pdf.py

Output:
    ./pdfs/<category-slug>/<article-slug>.pdf
    ./kb_articles.csv   (index of everything found, for reference/debugging)
"""

import csv
import re
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

BASE = "https://helpegypt.noon.com"
KB_HOME = f"{BASE}/portal/en/kb"
OUT_DIR = Path("../data")
CSV_PATH = Path("kb_articles.csv")

# Confirmed via devtools inspection of the live article page: the button is
# a <div role="button" aria-label="Download as PDF" title="Download as PDF"
# class="ArticleDetailLeftContainer__pdfView">, not a real <a href> link.
# It's a JS action (most Zoho Desk portals wire this to window.print() with
# a print-specific stylesheet), so we try to catch either a real file
# download OR a print dialog, and fall back to page.pdf() if neither fires.
PDF_BUTTON_SELECTORS = [
    "[aria-label='Download as PDF']",
    ".ArticleDetailLeftContainer__pdfView",
    "[title='Download as PDF']",
]


def slugify(url: str) -> str:
    path = urlparse(url).path.rstrip("/")
    return path.split("/")[-1] or "index"


def collect_category_links(page) -> list[str]:
    page.goto(KB_HOME, wait_until="networkidle")
    hrefs = page.eval_on_selector_all(
        "a[href*='/portal/en/kb/']",
        "els => els.map(e => e.href)",
    )
    cats = set()
    for h in hrefs:
        # category links look like /portal/en/kb/<slug>, not /portal/en/kb/articles/<slug>
        if "/portal/en/kb/" in h and "/articles/" not in h and h.rstrip("/") != KB_HOME:
            cats.add(h.split("?")[0])
    return sorted(cats)


def collect_article_links(page, category_url: str) -> list[str]:
    page.goto(category_url, wait_until="networkidle")
    # Some Zoho portals paginate or lazy-load; scroll a bit to trigger loads.
    for _ in range(5):
        page.mouse.wheel(0, 2000)
        page.wait_for_timeout(300)
    hrefs = page.eval_on_selector_all(
        "a[href*='/portal/en/kb/articles/']",
        "els => els.map(e => e.href)",
    )
    return sorted(set(h.split("?")[0] for h in hrefs))


def try_click_download_button(page, context, dest_path: Path, timeout_s: int = 30) -> bool:
    """
    Click the real 'Download as PDF' button and capture the actual PDF file.

    The response carries `Content-Disposition: attachment`, which makes
    Chromium hand it straight to its download manager instead of keeping it
    in the network buffer — so reading response.body() fails with
    "No resource with given identifier found". Instead we listen for
    Playwright's native 'download' event at the context level (covers every
    tab, not just the current page), which is how Chromium downloads are
    meant to be captured.
    """
    downloaded = {}

    def handle_download(download):
        downloaded["download"] = download

    context.on("download", handle_download)
    try:
        clicked = False
        for sel in PDF_BUTTON_SELECTORS:
            locator = page.locator(sel).first
            if locator.count() == 0:
                continue
            try:
                locator.click(timeout=3000)
                clicked = True
                break
            except Exception:
                continue

        if not clicked:
            return False

        # The site does an async generate -> poll status -> serve file
        # cycle, so the download can take a few seconds to actually start.
        deadline = time.time() + timeout_s
        while time.time() < deadline and "download" not in downloaded:
            page.wait_for_timeout(400)

        if "download" in downloaded:
            downloaded["download"].save_as(dest_path)
            return True
        return False
    finally:
        context.remove_listener("download", handle_download)


def print_to_pdf(page, dest_path: Path) -> None:
    """
    Render the article using the site's own print stylesheet — this is what
    the "Download as PDF" button actually triggers under the hood
    (window.print() with an @media print stylesheet), which doesn't work in
    headless mode on its own. Applying the print media type ourselves and
    calling page.pdf() reproduces the same output.
    """
    page.emulate_media(media="print")
    page.pdf(path=str(dest_path), format="A4", print_background=True)


def main():
    OUT_DIR.mkdir(exist_ok=True)
    rows = []

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        context = browser.new_context(accept_downloads=True)
        page = context.new_page()

        print("Collecting categories...")
        categories = collect_category_links(page)
        print(f"Found {len(categories)} categories")

        all_articles = []  # (category_slug, article_url)
        for cat_url in categories:
            cat_slug = slugify(cat_url)
            articles = collect_article_links(page, cat_url)
            print(f"  {cat_slug}: {len(articles)} articles")
            for a in articles:
                all_articles.append((cat_slug, a))

        # de-dup articles that appear in multiple categories
        seen = set()
        unique_articles = []
        for cat_slug, url in all_articles:
            if url not in seen:
                seen.add(url)
                unique_articles.append((cat_slug, url))

        print(f"\nTotal unique articles: {len(unique_articles)}\n")

        for i, (cat_slug, url) in enumerate(unique_articles, 1):
            art_slug = slugify(url)
            cat_dir = OUT_DIR / cat_slug
            cat_dir.mkdir(parents=True, exist_ok=True)
            dest = cat_dir / f"{art_slug}.pdf"

            print(f"[{i}/{len(unique_articles)}] {url}")
            try:
                page.goto(url, wait_until="networkidle")
                page.wait_for_timeout(500)

                method = "button"
                ok = try_click_download_button(page, context, dest)
                if not ok:
                    method = "print"
                    print_to_pdf(page, dest)

                title = page.title()
                rows.append({"category": cat_slug, "title": title, "url": url,
                             "pdf_path": str(dest), "method": method})
            except Exception as e:
                print(f"  FAILED: {e}")
                rows.append({"category": cat_slug, "title": "", "url": url,
                             "pdf_path": "", "method": f"error: {e}"})

            time.sleep(0.3)  # be polite

        browser.close()

    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["category", "title", "url", "pdf_path", "method"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone. Index written to {CSV_PATH}, PDFs under {OUT_DIR}/")


if __name__ == "__main__":
    main()