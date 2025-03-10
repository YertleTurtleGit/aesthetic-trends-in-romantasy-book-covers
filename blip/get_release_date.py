
import time
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.firefox import GeckoDriverManager

""" This script extracts the release date of books from Amazon product pages."""
""" Need this script because the HTML structure of the Amazon product pages was to complex to extract the release date with the other Metadata."""
""" After this script run the 'csv_merge.py' script to merge the CSV files."""


# Set up Selenium with Firefox
options = Options()
options.add_argument("--headless")  # Run in headless mode
service = Service(GeckoDriverManager().install())
driver = webdriver.Firefox(service=service, options=options)

# Read the CSV file that looks like this 'amazon_romantic_fantasy_main_*.csv'
file_path = 'amazon_romantic_fantasy_main_8.csv'
df = pd.read_csv(file_path, delimiter=',', encoding='utf-8-sig') 

# Create a new column for the release dates
df['Release Date'] = None

# Iterate through each URL in the CSV
for index, row in df.iterrows():
    url = row['URL']  # Get URL
    if pd.isna(url):  # Skip if URL is missing
        continue
    
    try:
        driver.get(url)  # Visit the URL
        WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.ID, "productSubtitle")))

        # Get the page source and parse it with BeautifulSoup
        soup = BeautifulSoup(driver.page_source, 'html.parser')

        # Extract the release date
        subtitle = soup.find('span', id='productSubtitle')
        if subtitle:
            release_date = subtitle.text.strip().split('–')[-1].strip()
            print(f"Release date: {release_date}")
            df.at[index, 'Release Date'] = release_date
        else:
            print(f"Release date not found for {url}")

    except Exception as e:
        print(f"Error processing {url}: {e}")

    time.sleep(2)  # Pause to avoid getting blocked

# Save the updated DataFrame to a new CSV file
# Change the output path to the desired file name (e.g., 'amazon_romantic_fantasy_with release_dates*.csv')
output_path = 'amazon_romantic_fantasy_with_release_dates_8.csv'  
df.to_csv(output_path, index=False, encoding='utf-8-sig')

# Close the browser
driver.quit()