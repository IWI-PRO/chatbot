import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from tqdm import tqdm
import json

def scrape_website(base_url, max_pages=5):
    visited = set()
    to_visit = [base_url]
    texts = []

    while to_visit and len(visited) < max_pages:
        url = to_visit.pop()
        if url in visited:
            continue
        visited.add(url)
        try:
            response = requests.get(url, timeout=5)
            soup = BeautifulSoup(response.text, 'html.parser')
            texts.append(soup.get_text(strip=True))

            for a_tag in soup.find_all('a', href=True):
                full_url = urljoin(url, a_tag['href'])
                if full_url.startswith(base_url):
                    to_visit.append(full_url)
        except Exception as e:
            print(f"[ERROR] {url}: {e}")
    return texts
