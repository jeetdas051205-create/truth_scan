"""
TruthScan AI - Real-World URL Scraping & Content Ingestion Engine
================================================================
Fetches and extracts clean editorial text from live web URLs:
- Realistic browser headers to bypass simple user-agent blocks
- Intelligent HTML extraction (<article>, <main>, <p>, OpenGraph metadata)
- Strips ads, navigation bars, footers, scripts, and styling
- Formats structured article payload for downstream inference
"""

import re
import requests
import bs4
from urllib.parse import urlparse

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Sec-Ch-Ua": '"Chromium";v="122", "Not(A:Brand";v="24", "Google Chrome";v="122"',
    "Sec-Ch-Ua-Mobile": "?0",
    "Sec-Ch-Ua-Platform": '"Windows"',
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Upgrade-Insecure-Requests": "1"
}


def scrape_article_from_url(url: str, timeout: int = 10) -> dict:
    """
    Downloads and extracts news article content from a live URL.
    Returns structured article dictionary or detailed error message.
    """
    if not url or not isinstance(url, str):
        return {
            "success": False,
            "error": "URL cannot be empty."
        }

    url = url.strip()
    if not url.startswith("http://") and not url.startswith("https://"):
        url = "https://" + url

    try:
        parsed_url = urlparse(url)
        domain = parsed_url.netloc.replace("www.", "")
    except Exception:
        return {
            "success": False,
            "error": "The provided URL is not formatted correctly."
        }

    try:
        response = requests.get(url, headers=HEADERS, timeout=timeout, allow_redirects=True)
        if response.status_code != 200:
            return {
                "success": False,
                "error": f"Unable to fetch URL (HTTP {response.status_code}). The website may require login or subscription access."
            }

        html = response.text
        soup = bs4.BeautifulSoup(html, "html.parser")

        # 1. Clean out irrelevant tags
        for unwanted in soup(["script", "style", "nav", "footer", "header", "aside", "form", "svg"]):
            unwanted.extract()

        # 2. Extract Title
        title = ""
        og_title = soup.find("meta", property="og:title")
        if og_title and og_title.get("content"):
            title = og_title["content"].strip()
        elif soup.title and soup.title.string:
            title = soup.title.string.strip()
        elif soup.find("h1"):
            title = soup.find("h1").get_text().strip()
        else:
            title = f"News Article from {domain}"

        # 3. Extract Site Name / Publisher
        site_name = domain
        og_site = soup.find("meta", property="og:site_name")
        if og_site and og_site.get("content"):
            site_name = og_site["content"].strip()

        # 4. Extract Article Paragraphs
        article_elem = soup.find("article") or soup.find("main") or soup
        paragraphs = []
        for p in article_elem.find_all("p"):
            p_text = p.get_text().strip()
            # Filter out copyright notices, disclaimers, or cookie notices
            if len(p_text.split()) >= 6 and not re.search(r'cookie|privacy policy|terms of service|copyright \d{4}', p_text, re.IGNORECASE):
                paragraphs.append(p_text)

        full_body = "\n\n".join(paragraphs)
        words = full_body.split()

        if len(words) < 25:
            # Fallback to general text extraction
            all_text = soup.get_text(separator=' ')
            clean_lines = [l.strip() for l in all_text.split('\n') if len(l.strip().split()) >= 8]
            full_body = "\n\n".join(clean_lines[:15])
            words = full_body.split()

        if len(words) < 20:
            return {
                "success": False,
                "error": "The target webpage did not contain sufficient textual content. It may be heavily media-focused or client-side rendered with JavaScript.",
                "title": title,
                "domain": domain
            }

        word_count = len(words)
        read_time_min = max(1, round(word_count / 200))

        # Take first ~600 words for optimal neural inference
        inference_text = " ".join(words[:600])

        return {
            "success": True,
            "url": url,
            "domain": domain,
            "site_name": site_name,
            "title": title,
            "full_body": full_body,
            "inference_text": inference_text,
            "word_count": word_count,
            "read_time_min": read_time_min,
            "excerpt": " ".join(words[:45]) + "..."
        }

    except requests.exceptions.Timeout:
        return {
            "success": False,
            "error": f"Connection to {domain} timed out after {timeout} seconds. The host server may be slow or blocking automated requests."
        }
    except requests.exceptions.ConnectionError:
        return {
            "success": False,
            "error": f"Could not establish network connection to {domain}. Please verify the URL or check your internet connection."
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"An error occurred while parsing the webpage: {str(e)}"
        }

