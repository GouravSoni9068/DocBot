import json
import os
from dotenv import load_dotenv
from src.scraper import DocumentationScraper

# Load environment variables
load_dotenv()

# Load URLs from file
with open('data/urls.json', 'r') as f:
    urls = json.load(f)

# Initialize scraper with authentication
scraper = DocumentationScraper(
    urls[0],
    username=os.getenv('ATLASSIAN_USERNAME'),
    api_token=os.getenv('ATLASSIAN_API_TOKEN')
)

# Scrape documentation
print("Starting documentation scraping...")
scraper.scrape_documentation(urls)

# Save scraped data
scraper.save_to_json('data/scraped_docs.json')
print("Documentation saved to data/scraped_docs.json")
