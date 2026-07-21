import asyncio
import logging
from datetime import datetime, timezone
from urllib.parse import urljoin, urlparse

from playwright.async_api import async_playwright

from graphitti.config import MAX_DEPTH, MAX_PAGES, REQUEST_TIMEOUT_MS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("crawler")


def _same_domain(url: str, root_domain: str) -> bool:
    try:
        return urlparse(url).netloc == root_domain
    except Exception:
        return False


def _clean_url(base: str, href: str) -> str | None:
    if not href or href.startswith(("mailto:", "tel:", "javascript:", "#")):
        return None
    absolute = urljoin(base, href)
    parsed = urlparse(absolute)
    if parsed.scheme not in ("http", "https"):
        return None
    return parsed._replace(fragment="").geturl()


async def _extract_page(page, url: str, depth: int) -> dict:
    title = await page.title()

    text = await page.evaluate(
        """() => {
            const kill = [
                'script', 'style', 'nav', 'footer', 'noscript', 'svg',
                // reference/citation/bibliography sections — kept separately
                // from the base kill list above so it's obvious why these
                // exist: this is what was letting footnote/ISBN/"Retrieved
                // on ..." boilerplate through into extraction & retrieval.
                'ol.references', '.reflist', '.mw-references-wrap',
                'sup.reference', '.citation', '[role="doc-endnotes"]',
                '[role="doc-bibliography"]', '.navbox', '.catlinks',
            ];
            const clone = document.body.cloneNode(true);
            kill.forEach(tag => clone.querySelectorAll(tag).forEach(el => el.remove()));
            return clone.innerText;
        }"""
    )
    text = " ".join(text.split())

    links = await page.eval_on_selector_all(
        "a[href]", "els => els.map(e => e.getAttribute('href'))"
    )

    meta_description = ""
    try:
        meta_description = await page.get_attribute(
            "meta[name='description']", "content"
        ) or ""
    except Exception:
        pass

    return {
        "url": url,
        "title": title,
        "text": text,
        "meta_description": meta_description,
        "links": links,
        "depth": depth,
        "crawled_at": datetime.now(timezone.utc).isoformat(),
    }


async def crawl(
    start_url: str,
    max_depth: int = MAX_DEPTH,
    max_pages: int = MAX_PAGES,
    same_domain_only: bool = True,
) -> list[dict]:
    root_domain = urlparse(start_url).netloc
    visited: set[str] = set()
    queue: list[tuple[str, int]] = [(start_url, 0)]
    pages: list[dict] = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent="Graphitti-Crawler/1.0 (+https://example.com/bot)"
        )
        page = await context.new_page()
        page.set_default_timeout(REQUEST_TIMEOUT_MS)

        while queue and len(pages) < max_pages:
            url, depth = queue.pop(0)
            if url in visited or depth > max_depth:
                continue
            visited.add(url)

            try:
                log.info(f"Crawling (depth={depth}): {url}")
                await page.goto(url, wait_until="domcontentloaded")
                data = await _extract_page(page, url, depth)
            except Exception as e:
                log.warning(f"Failed {url}: {e}")
                continue

            pages.append(data)

            if depth < max_depth:
                for href in data["links"]:
                    nxt = _clean_url(url, href)
                    if not nxt or nxt in visited:
                        continue
                    if same_domain_only and not _same_domain(nxt, root_domain):
                        continue
                    queue.append((nxt, depth + 1))

        await browser.close()

    log.info(f"Crawl finished: {len(pages)} pages")
    return pages


def crawl_sync(start_url: str, max_depth: int = MAX_DEPTH, max_pages: int = MAX_PAGES) -> list[dict]:
    return asyncio.run(crawl(start_url, max_depth, max_pages))
