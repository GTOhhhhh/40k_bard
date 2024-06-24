# 40k_bard
RAG Pipeline for 40k story inference using scraped lore

## Scraper Usage

1. Install the required Python packages using pip:

2. Run the scraper using the following command:


The scraper will download all pages from the Warhammer 40K Lexicanum and save them as JSON files in the `wh40k_pages` directory.

You can specify the number of pages to scrape and the output directory using command line arguments. For example, to scrape 100 pages and save them in a directory named `my_output_dir`, you can use the following command:
```bash
python scraper.py -n 100 -o my_output_dir
```

The scraper will create the output directory if it doesn't exist, and it will overwrite any existing files with the same name.

The scraper will process and store the documents in the PostgreSQL database. You can then use the `vector_db.py` script to query the database.