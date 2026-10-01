import requests
from bs4 import BeautifulSoup
from typing import Dict, Any

def retrieve_webpage(url: str, max_chars: int = 4000) -> Dict[str, Any]:
    """
    Fetches clean body text content from a target webpage URL.
    Returns dict: {'url': str, 'title': str, 'content': str, 'status': str}
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    try:
        response = requests.get(url, headers=headers, timeout=8)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Remove scripts, styles, header, nav, footer
        for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "svg"]):
            tag.decompose()

        title = soup.title.string.strip() if soup.title and soup.title.string else url

        # Extract text paragraphs
        paragraphs = [p.get_text().strip() for p in soup.find_all(["p", "h1", "h2", "h3", "li"]) if p.get_text().strip()]
        content = "\n".join(paragraphs)

        if len(content) > max_chars:
            content = content[:max_chars] + "... [content truncated]"

        if not content:
            content = "No visible paragraph content extracted from this URL."

        return {
            "url": url,
            "title": title,
            "content": content,
            "status": "success"
        }
    except Exception as e:
        return {
            "url": url,
            "title": "Error Retrieving Webpage",
            "content": f"Unable to fetch content from {url}: {str(e)}",
            "status": "error"
        }
