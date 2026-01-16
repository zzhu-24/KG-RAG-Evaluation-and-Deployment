import argparse
import json
import re
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


BASE_URL = "https://www.formula1.com"
LIST_URL = "https://www.formula1.com/en/latest"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; ResearchScraper/1.0; +https://example.com/bot)"
}

ARTICLE_HREF_RE = re.compile(r"^/en/latest/article/[^?#]+", re.IGNORECASE)


def parse_published_time(published_time: Optional[str]) -> Optional[datetime]:
    """
    Parse published time string to datetime object.
    Supports multiple common formats: ISO 8601, RFC 2822, etc.
    """
    if not published_time:
        return None
    
    # Common datetime formats
    formats = [
        "%Y-%m-%dT%H:%M:%S%z",  # ISO 8601 with timezone
        "%Y-%m-%dT%H:%M:%S.%f%z",  # ISO 8601 with microseconds and timezone
        "%Y-%m-%dT%H:%M:%S",  # ISO 8601 without timezone
        "%Y-%m-%dT%H:%M:%S.%f",  # ISO 8601 with microseconds
        "%Y-%m-%d %H:%M:%S",  # Standard format
        "%Y-%m-%d",  # Date only
    ]
    
    # Remove colons in timezone info (e.g., +08:00 -> +0800)
    cleaned = re.sub(r"([+-])(\d{2}):(\d{2})$", r"\1\2\3", published_time)
    
    for fmt in formats:
        try:
            return datetime.strptime(cleaned, fmt)
        except ValueError:
            continue
    
    # If all formats fail, try using dateutil (if available)
    try:
        from dateutil import parser as date_parser
        return date_parser.parse(published_time)
    except ImportError:
        pass
    except Exception:
        pass
    
    return None


def get_week_start(date: datetime) -> datetime:
    """
    Get the start date of the week (Monday) for the given date.
    If date is None, return the start date of the current week.
    """
    if date is None:
        date = datetime.now()
    # Calculate offset to Monday (0=Monday, 6=Sunday)
    days_since_monday = date.weekday()
    week_start = date - timedelta(days=days_since_monday)
    # Return 00:00:00 of that day
    return week_start.replace(hour=0, minute=0, second=0, microsecond=0)


def get_week_filename(week_start: datetime, output_dir: Path) -> str:
    """
    Generate filename based on week start date.
    Format: f1_news_YYYY-MM-DD.jsonl
    """
    filename = f"f1_news_{week_start.strftime('%Y-%m-%d')}.jsonl"
    return str(output_dir / filename)


def fetch_soup(url: str, session: requests.Session) -> BeautifulSoup:
    """Fetch HTML and parse with BeautifulSoup."""
    resp = session.get(url, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")


def extract_article_urls_from_list(page_url: str, session: requests.Session) -> List[str]:
    """
    Extract article URLs from /en/latest?page=X list page.
    Robustly scans all <a href="..."> and filters /en/latest/article/... links.
    """
    soup = fetch_soup(page_url, session)

    urls = set()
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if ARTICLE_HREF_RE.match(href):
            urls.add(urljoin(BASE_URL, href))

    return sorted(urls)


def _try_parse_json_ld_newsarticle(soup: BeautifulSoup) -> Optional[Dict]:
    """
    Try to parse NewsArticle-like JSON-LD blocks for headline/datePublished/articleBody/description.
    """
    for tag in soup.find_all("script", attrs={"type": "application/ld+json"}):
        raw = tag.string
        if not raw:
            continue
        raw = raw.strip()
        try:
            data = json.loads(raw)
        except Exception:
            continue

        # JSON-LD may be a dict, list, or contain @graph
        candidates = []
        if isinstance(data, dict):
            if "@graph" in data and isinstance(data["@graph"], list):
                candidates.extend(data["@graph"])
            else:
                candidates.append(data)
        elif isinstance(data, list):
            candidates.extend(data)

        for obj in candidates:
            if not isinstance(obj, dict):
                continue
            typ = obj.get("@type")
            # sometimes @type is list
            if isinstance(typ, list):
                is_news = any(t in ("NewsArticle", "Article") for t in typ)
            else:
                is_news = typ in ("NewsArticle", "Article")

            if not is_news:
                continue

            return {
                "title": obj.get("headline") or obj.get("name"),
                "summary": obj.get("description"),
                "published_time": obj.get("datePublished") or obj.get("dateCreated"),
                "content": obj.get("articleBody"),
            }

    return None


def _meta_content(soup: BeautifulSoup, key: str, attr: str = "property") -> Optional[str]:
    tag = soup.find("meta", attrs={attr: key})
    if tag and tag.get("content"):
        return tag["content"].strip()
    return None


def extract_article_text_fallback(soup: BeautifulSoup) -> str:
    """
    Fallback: extract readable article text from <main> by collecting headings/paragraphs,
    stopping before common tail sections.
    """
    main = soup.find("main")
    if not main:
        main = soup  # last resort

    stop_markers = {
        "Related Articles",
        "OUR PARTNERS",
        "Next Up",
    }

    chunks: List[str] = []

    # Collect text from typical content tags in reading order
    for el in main.find_all(["h2", "h3", "p", "li"], recursive=True):
        text = el.get_text(" ", strip=True)
        if not text:
            continue

        if text in stop_markers:
            break

        # Avoid obvious navigation / cookie / menu fragments if any slip in
        if text.lower() in {"skip to content", "menu"}:
            continue

        chunks.append(text)

    # De-duplicate consecutive identical lines
    deduped: List[str] = []
    for t in chunks:
        if not deduped or deduped[-1] != t:
            deduped.append(t)

    return "\n".join(deduped).strip()


def parse_article(url: str, session: requests.Session) -> Dict:
    soup = fetch_soup(url, session)

    # 1) Prefer JSON-LD (often best quality)
    ld = _try_parse_json_ld_newsarticle(soup) or {}

    # 2) Fill missing fields from meta tags / HTML
    title = ld.get("title")
    if not title:
        h1 = soup.find("h1")
        title = h1.get_text(strip=True) if h1 else None
    if not title:
        title = _meta_content(soup, "og:title")

    summary = ld.get("summary") or _meta_content(soup, "og:description") or _meta_content(soup, "description", attr="name")

    published_time = ld.get("published_time") or _meta_content(soup, "article:published_time")
    if not published_time:
        time_tag = soup.find("time")
        if time_tag and time_tag.get("datetime"):
            published_time = time_tag["datetime"].strip()

    content = ld.get("content")
    if not content or not content.strip():
        content = extract_article_text_fallback(soup)

    return {
        "title": title,
        "summary": summary,
        "published_time": published_time,
        "url": url,
        "content": content,
    }


def run(start_page: int, end_page: int, sleep_sec: float, out_dir: str):
    """
    抓取 F1 新闻并按周分组保存到不同文件，同时进行去重。
    
    Args:
        start_page: 起始页码
        end_page: 结束页码
        sleep_sec: 请求间隔（秒）
        out_dir: 输出目录，将在此目录下创建按周分组的文件
    """
    assert 1 <= start_page <= end_page <= 100

    # 创建输出目录
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 用于去重的集合（基于 URL）
    seen_urls: set[str] = set()
    
    # 按周分组的新闻字典：{week_start: [items]}
    news_by_week: Dict[datetime, List[Dict]] = defaultdict(list)

    with requests.Session() as session:
        # 1) Collect URLs from list pages
        all_urls: List[str] = []
        for p in tqdm(range(start_page, end_page + 1), desc="Collecting list pages"):
            page_url = f"{LIST_URL}?page={p}"
            try:
                urls = extract_article_urls_from_list(page_url, session)
                all_urls.extend(urls)
            except Exception as e:
                print(f"[WARN] list page failed: page={p} err={e}")
            time.sleep(sleep_sec)

        # Unique URLs
        all_urls = sorted(set(all_urls))

        # 2) Scrape articles and group by week
        for url in tqdm(all_urls, desc="Scraping articles"):
            # 去重检查
            if url in seen_urls:
                continue
            
            try:
                item = parse_article(url, session)
                # Keep only valid items
                if not item["title"] or not item["content"]:
                    continue
                
                # 标记为已处理
                seen_urls.add(url)
                
                # 解析发布时间并确定所属周
                published_time = parse_published_time(item.get("published_time"))
                if published_time is None:
                    # 如果没有发布时间，使用当前时间
                    published_time = datetime.now()
                    item["published_time"] = published_time.isoformat()
                
                week_start = get_week_start(published_time)
                news_by_week[week_start].append(item)
                
            except Exception as e:
                print(f"[WARN] article failed: url={url} err={e}")
            
            time.sleep(sleep_sec)

        # 3) Write articles grouped by week
        print(f"\n[INFO] Writing {len(news_by_week)} week files...")
        for week_start in sorted(news_by_week.keys()):
            items = news_by_week[week_start]
            filename = get_week_filename(week_start, output_dir)
            
            # 按发布时间排序
            items.sort(key=lambda x: parse_published_time(x.get("published_time")) or datetime.min)
            
            with open(filename, "w", encoding="utf-8") as f:
                for item in items:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
            print(f"[INFO] Wrote {len(items)} articles to {filename} (week starting {week_start.strftime('%Y-%m-%d')})")
        
        total_articles = sum(len(items) for items in news_by_week.values())
        print(f"[INFO] Total: {total_articles} unique articles across {len(news_by_week)} weeks")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=1, help="Start page (1-100)")
    parser.add_argument("--end", type=int, default=5, help="End page (1-100)")
    parser.add_argument("--sleep", type=float, default=1.2, help="Sleep seconds between requests")
    parser.add_argument("--out-dir", type=str, default="news", help="Output directory for weekly JSONL files")
    args = parser.parse_args()

    run(args.start, args.end, args.sleep, args.out_dir)
