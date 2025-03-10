import os
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.firefox import GeckoDriverManager

# Funktion, um die URL für jede Seite zu erstellen
def get_page_url(base_url, page_number):
    return base_url + f"&page={page_number}"

def scrape_main_info(item):
    try:
        print("Processing main info...")
        
        title = item.find('h2').text.strip() if item.find('h2') else 'No Title'
        
        atag = item.find('a', {'class': 'a-link-normal'})
        url = 'https://amazon.de' + atag.get('href') if atag else 'No URL'
        
        price_element = item.find('span', {'class': 'a-price'})
        price = price_element.find('span', {'class': 'a-offscreen'}).text.strip() if price_element else 'No Price'
        
        rating_element = item.find('span', {'class': 'a-icon-alt'})
        rating = rating_element.text.strip() if rating_element else 'No Rating'
        
        review_count_element = item.find('span', {
            'class': ['a-size-mini', 'a-color-base', 'puis-light-weight-text'],
            'aria-label': lambda x: x and 'Bewertungen' in str(x)
        })
        review_count = review_count_element.text.strip().strip('()').strip() if review_count_element else 'No Reviews'
        
        author = 'No Author'
        author_div = item.find('div', {'class': 'a-row'})
        if author_div:
            author_links = author_div.find_all('a', {'class': 'a-link-normal s-underline-text s-underline-link-text s-link-style'})
            if author_links:
                authors = [link.text.strip() for link in author_links]
                author = ' und '.join(authors)
        
        image_element = item.find('img', {'class': 's-image'})
        image_url = image_element.get('src') if image_element else 'No Image'
        
        return {
            'title': title,
            'price': price,
            'rating': rating,
            'review_count': review_count,
            'author': author,
            'url': url,
            'image_url': image_url
        }
    except Exception as e:
        print(f"Error scraping main info: {e}")
        return None

def scrape_release_date(item):
    try:
        print("Processing release date...")
        title = item.find('h2').text.strip() if item.find('h2') else 'No Title'
        
        release_date = 'No Release Date'
        author_div = item.find('div', {'class': 'a-row'})
        if author_div:
            # Look for the last span that contains the date (after the | separator)
            spans = author_div.find_all('span', {'class': 'a-size-base a-color-secondary a-text-normal'})
            if spans:  # Get the last span which should be the date
                release_date = spans[-1].text.strip()
            else:
                # Backup method: look for any span after a separator
                all_spans = author_div.find_all('span')
                for span in all_spans:
                    if '|' in span.text:
                        next_span = span.find_next_sibling('span')
                        if next_span:
                            release_date = next_span.text.strip()
                            break
        
        return {
            'title': title,
            'release_date': release_date
        }
    except Exception as e:
        print(f"Error scraping release date: {e}")
        return None

def scrape_amazon(base_url, max_pages=10, max_retries=3):
    records_main = []
    records_release = []
    current_page = 1
    
    while current_page <= max_pages:
        retries = 0
        success = False
        
        while retries < max_retries and not success:
            try:
                print(f"\nAttempting to scrape page {current_page} (Attempt {retries + 1}/{max_retries})")
                
                # Initialize new driver for each retry
                ua = UserAgent()
                options = Options()
                options.set_preference("general.useragent.override", ua.random)
                options.headless = True
                
                service = Service(GeckoDriverManager().install())
                driver = webdriver.Firefox(service=service, options=options)
                
                # Set page load timeout
                driver.set_page_load_timeout(30)
                
                # Get the page
                url = get_page_url(base_url, current_page)
                driver.get(url)
                
                # Wait for content to load
                try:
                    WebDriverWait(driver, 20).until(
                        EC.presence_of_element_located((By.ID, 'search'))
                    )
                    
                    # Parse the page
                    soup = BeautifulSoup(driver.page_source, 'html.parser')
                    results = soup.find_all('div', {'data-component-type': 's-search-result'})
                    
                    # Process results
                    page_records_main = []
                    page_records_release = []
                    
                    for item in results:
                        main_info = scrape_main_info(item)
                        if main_info:
                            page_records_main.append(main_info)
                        
                        release_info = scrape_release_date(item)
                        if release_info:
                            page_records_release.append(release_info)
                    
                    # Only append records if we found any
                    if page_records_main or page_records_release:
                        records_main.extend(page_records_main)
                        records_release.extend(page_records_release)
                        print(f"Successfully scraped {len(page_records_main)} items from page {current_page}")
                        success = True
                    else:
                        print(f"No results found on page {current_page}")
                        raise Exception("No results found")
                    
                except Exception as e:
                    print(f"Error processing page content: {str(e)}")
                    raise
                    
            except Exception as e:
                print(f"Error on attempt {retries + 1}: {str(e)}")
                retries += 1
                time.sleep(5 * retries)  # Incremental delay between retries
            
            finally:
                try:
                    driver.quit()
                except:
                    pass
        
        if not success:
            print(f"Failed to scrape page {current_page} after {max_retries} attempts")
            # Optional: decide whether to continue to next page or stop entirely
            if len(records_main) == 0:
                print("No data collected yet, stopping scraper")
                break
        
        current_page += 1
        time.sleep(10)  # Wait between pages
    
    if len(records_main) == 0:
        raise Exception("No data was collected")
    
    # Create DataFrames
    main_df = pd.DataFrame(records_main)
    release_df = pd.DataFrame(records_release)
    
    # Merge the DataFrames on title
    merged_df = pd.merge(main_df, release_df[['title', 'release_date']], on='title', how='left')
    
    print(f"\nTotal items collected: {len(merged_df)}")
    return merged_df
# Funktion, um Bilder herunterzuladen und zu speichern
def save_images(df, folder='images'):
    if not os.path.exists(folder):
        os.makedirs(folder)

    for index, row in df.iterrows():
        image_url = row['Image URL']
        if image_url:
            # Dateiname bereinigen (max. 50 Zeichen, keine Sonderzeichen)
            image_name = f"{''.join(c for c in row['Title'][:50] if c.isalnum() or c in (' ', '_')).replace(' ', '_')}.jpg"
            image_path = os.path.join(folder, image_name)
            try:
                response = requests.get(image_url, stream=True)
                if response.status_code == 200:
                    with open(image_path, 'wb') as f:
                        for chunk in response.iter_content(1024):
                            f.write(chunk)
                    print(f"Image saved: {image_path}")
                    df.at[index, 'Local Image Path'] = image_path
                else:
                    print(f"Error downloading image: {image_url}")
                    df.at[index, 'Local Image Path'] = 'Failed'
            except Exception as e:
                print(f"Error with {image_url}: {e}")
                df.at[index, 'Local Image Path'] = 'Failed'

    return df

if __name__ == "__main__":
    base_url = "https://www.amazon.de/s?k=b%C3%BCcher+romantische+fantasy&i=stripbooks&rh=n%3A142"
    
    try:
        print("Starting to scrape the Amazon page...")
        df = scrape_amazon(base_url, max_pages=10)
        print("Data successfully collected.")
        
        print("Downloading images...")
        df = save_images(df)
        print("Images successfully saved.")
        
        output_file = 'amazon_romantic_fantasy.csv'
        df.to_csv(output_file, index=False, encoding="utf-8-sig")
        print(f"File exported: {output_file}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Attempting to save any collected data...")
        try:
            if 'df' in locals() and not df.empty:
                emergency_file = 'amazon_romantic_fantasy_partial.csv'
                df.to_csv(emergency_file, index=False, encoding="utf-8-sig")
                print(f"Partial data saved to {emergency_file}")
        except:
            print("Could not save partial data")