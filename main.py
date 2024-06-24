import asyncio
import aiohttp
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import json
import os
from tqdm import tqdm
import argparse

BASE_URL = "https://wh40k.lexicanum.com"
ALL_PAGES_URL = f"{BASE_URL}/wiki/Special:AllPages"
OUTPUT_DIR = "wh40k_pages"

async def get_pages(session, max_pages):
    all_pages = []
    current_url = ALL_PAGES_URL
    
    while current_url and (max_pages is None or len(all_pages) < max_pages):
        async with session.get(current_url) as response:
            soup = BeautifulSoup(await response.text(), 'html.parser')
            
            # Extract page links
            pages = soup.select('ul.mw-allpages-chunk li a')
            all_pages.extend([urljoin(BASE_URL, page['href']) for page in pages])
            
            # Break if we have enough pages
            if max_pages and len(all_pages) >= max_pages:
                break
            
            # Find next page link
            next_page = soup.select_one('a:contains("Next page")')
            current_url = urljoin(BASE_URL, next_page['href']) if next_page else None
    
    return all_pages[:max_pages] if max_pages else all_pages

async def process_page(session, url):
    async with session.get(url) as response:
        soup = BeautifulSoup(await response.text(), 'html.parser')
        
        # Extract title
        title = soup.select_one('#firstHeading').text.strip()
        
        # Extract main content
        content = soup.select_one('#mw-content-text')
        if content:
            # Remove unwanted elements
            for unwanted in content.select('.mw-editsection, .magnify, .noprint'):
                unwanted.decompose()
            
            # Extract text content
            text_content = ' '.join(p.text for p in content.select('p'))
        else:
            text_content = ""
        
        return {
            "title": title,
            "url": url,
            "content": text_content
        }

async def save_page(page_data, output_dir):
    # Create a filename from the title
    filename = f"{page_data['title'].replace(' ', '_')}.json"
    filepath = os.path.join(output_dir, filename)
    
    # Save the page data as JSON
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(page_data, f, ensure_ascii=False, indent=2)

async def main(max_pages, output_dir):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    async with aiohttp.ClientSession() as session:
        pages = await get_pages(session, max_pages)
        print(f"Found {len(pages)} pages to process")
        
        # Process pages with a progress bar
        for page_url in tqdm(pages, desc="Processing pages"):
            page_data = await process_page(session, page_url)
            await save_page(page_data, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape Warhammer 40K Lexicanum pages")
    parser.add_argument("-n", "--num_pages", type=int, default=None, help="Number of pages to scrape. If not specified, scrapes all pages.")
    parser.add_argument("-o", "--output", type=str, default=OUTPUT_DIR, help="Output directory for scraped pages")
    args = parser.parse_args()

    asyncio.run(main(args.num_pages, args.output))