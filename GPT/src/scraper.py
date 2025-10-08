import requests
from bs4 import BeautifulSoup
import json
from typing import Dict, List
import os

class DocumentationScraper:
    def __init__(self, base_url: str, username: str = None, api_token: str = None):
        self.base_url = base_url
        self.documents = []
        self.auth = (username, api_token) if username and api_token else None

    def scrape_page(self, url: str) -> Dict:
        """Scrape content from a single page."""
        try:
            response = requests.get(url, auth=self.auth)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract main content (adjust selectors based on actual website structure)
            title = soup.find('h1').text.strip() if soup.find('h1') else ""
            content = []
            
            # Get all paragraphs and headers
            main_content = soup.find('main') or soup.find('article') or soup
            for element in main_content.find_all(['p', 'h2', 'h3', 'h4', 'li']):
                text = element.get_text().strip()
                if text:
                    content.append(text)
            
            return {
                'url': url,
                'title': title,
                'content': '\n'.join(content)
            }
        except Exception as e:
            print(f"Error scraping {url}: {str(e)}")
            return None

    def scrape_documentation(self, urls: List[str]) -> None:
        """Scrape multiple documentation pages."""
        for url in urls:
            doc = self.scrape_page(url)
            if doc:
                self.documents.append(doc)

    def save_to_json(self, output_file: str = 'docs_data.json') -> None:
        """Save scraped documentation to a JSON file."""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.documents, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    # Example usage
    base_url = "https://example-docs-url.com"  # Replace with actual docs URL
    urls_to_scrape = [
        f"{base_url}/page1",
        f"{base_url}/page2",
    ]
    
    scraper = DocumentationScraper(base_url)
    scraper.scrape_documentation(urls_to_scrape)
    scraper.save_to_json('data/scraped_docs.json')
