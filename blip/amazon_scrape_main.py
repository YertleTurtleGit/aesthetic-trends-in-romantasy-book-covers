
import time
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

""" After this script run the 'get_release_date.py' script to get the release dates of the books."""

def get_page_url(base_url, page_number):
    return base_url + f"&page={page_number}"

def clean_text(text):
    if not text:
        return None
    return ' '.join(text.strip().split())

def extract_price(price_text):
    if not price_text:
        return None
    return price_text.replace('€', '').strip()

def extract_rating(rating_text):
    if not rating_text:
        return None
    try:
        return float(rating_text.split()[0].replace(',', '.'))
    except:
        return None

def extract_review_count(review_text):
    if not review_text:
        return None
    try:
        return int(''.join(filter(str.isdigit, review_text)))
    except:
        return None

def scrape_main_data(item, current_id):
    """Scrape main book data including author"""
    try:
        # Title
        title_element = item.find('h2')
        title = clean_text(title_element.text) if title_element else None
        
        # URL
        atag = item.find('a', {'class': 'a-link-normal'})
        url = 'https://amazon.de' + atag.get('href') if atag else None
        
        # Price
        price_element = item.find('span', {'class': 'a-price'})
        if price_element:
            price = extract_price(price_element.find('span', {'class': 'a-offscreen'}).text)
        else:
            price = None
            
        # Rating
        rating_element = item.find('span', {'class': 'a-icon-alt'})
        rating = extract_rating(rating_element.text) if rating_element else None
        
        # Review Count
        review_element = item.find('span', {
            'class': ['a-size-mini', 'a-color-base', 'puis-light-weight-text'],
            'aria-label': lambda x: x and 'Bewertungen' in str(x)
        })
        review_count = extract_review_count(review_element.text) if review_element else None
        
        # Author
        author = None
        author_div = item.find('div', {'class': 'a-row a-size-mini a-color-secondary'})
        if author_div:
            spans = author_div.find_all('span', {'class': 'a-size-mini puis-light-weight-text'})
            if len(spans) >= 2:
                author = clean_text(spans[1].text)
        
        # Image URL
        image_element = item.find('img', {'class': 's-image'})
        image_url = image_element.get('src') if image_element else None
        
        return {
            'ID': current_id,
            'Title': title,
            'Author': author,
            'Price': price,
            'Rating': rating,
            'Review_Count': review_count,
            'URL': url,
            'Image_URL': image_url
        }
        
    except Exception as e:
        print(f"Error scraping main data: {e}")
        return None

# 
def scrape_amazon(base_url, max_pages=400):   
    """Scrape Amazon search results for book data"""

    # Set up the web driver  
    ua = UserAgent()
    options = Options()
    options.set_preference("general.useragent.override", ua.random)
    options.headless = True

    service = Service(GeckoDriverManager().install())
    driver = webdriver.Firefox(service=service, options=options)

    main_data_list = []
    current_id = 1

    # Loop through each page of the search results
    for page in range(1, max_pages + 1):
        print(f"Scraping page {page}...")
        url = get_page_url(base_url, page)
        
        # Load the page
        try:
            driver.get(url)
            WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.ID, 'search')))
            
            # Parse the HTML
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            results = soup.find_all('div', {'data-component-type': 's-search-result'})

            # Scrape the main data for each book
            for item in results:
                main_data = scrape_main_data(item, current_id)
                if main_data and main_data['Title']:
                    main_data_list.append(main_data)
                    current_id +=1 
            
            # Sleep to avoid getting blocked
            time.sleep(3)
            
        except Exception as e:
            print(f"Error on page {page}: {e}")
            continue
    
    # Close the web driver
    driver.quit()

    # Create separate DataFrames
    main_df = pd.DataFrame(main_data_list)
    
    return main_df

if __name__ == "__main__":
    """Main script to scrape Amazon book data"""
    # Different base URLs for different searches to get as much data as possible

    
    #base_url = "https://www.amazon.de/s?k=romantische+fantasy+b%C3%BCcher&i=stripbooks&rh=n%3A186606%2Cp_n_binding_browse-bin%3A492558011%257C492559011%257C9330699031&dc&__mk_de_DE=%C3%85M%C3%85%C5%BD%C3%95%C3%91&crid=ZAVTYCQ9TR1&qid=1736256613&rnid=492557011&sprefix=romantische+fantasy+b%C3%BCcher+%2Cstripbooks%2C106&xpid=IlmEest6TXqY0&ref=sr_pg_1"
    # base_url_1 = "https://www.amazon.de/s?k=Fantasy+Liebesromane&i=stripbooks&rh=n%3A16381996031%2Cp_n_binding_browse-bin%3A492558011%257C492559011%257C9330699031&dc&__mk_de_DE=%C3%85M%C3%85%C5%BD%C3%95%C3%91&qid=1736263909&rnid=492557011&ref=sr_nr_p_n_binding_browse-bin_3&ds=v1%3A3jMddlQ7XOF7CBoDA5hsZSMnnwWtejssLqsCPPqyC9E"
    # print("Starting to scrape Amazon...")
    # df = scrape_amazon(base_url_1, max_pages=25)
    # base_url_2 = "https://www.amazon.de/s?k=Fantasy+Liebesromane&i=stripbooks&rh=n%3A16381993031%2Cp_n_binding_browse-bin%3A492558011%257C492559011%257C9330699031&dc&ds=v1%3AdMwplg8cKrLdzBaWkbuISfjbdaCQASxG8E7BHpzK6wY&__mk_de_DE=%C3%85M%C3%85%C5%BD%C3%95%C3%91&qid=1736263932&rnid=492557011&ref=sr_nr_p_n_binding_browse-bin_3"
    # print("Starting to scrape Amazon...")
    # df_2= scrape_amazon(base_url_2, max_pages=178)
    # base_url_3 = "https://www.amazon.de/s?k=Fantasy+Liebesromane&i=stripbooks&rh=n%3A16381998031%2Cp_n_binding_browse-bin%3A492558011%257C492559011%257C9330699031&dc&__mk_de_DE=%C3%85M%C3%85%C5%BD%C3%95%C3%91&qid=1736263954&rnid=492557011&ref=sr_nr_p_n_binding_browse-bin_3&ds=v1%3AHUsa4va0CKaXyDZDUS0Hj7MwX1JcWm8eVds9kl30pbo"
    # print("Starting to scrape Amazon...")
    # df_3 = scrape_amazon(base_url_3, max_pages=36)
    # base_url_4 = "https://www.amazon.de/s?i=stripbooks&bbn=186606&rh=n%3A86208402031%2Cp_n_feature_three_browse-bin%3A15425222031&s=popularity-rank&dc&fs=true&qid=1736266285&rnid=4192708031&xpid=BapcAeWRXzoze&ref=sr_nr_p_n_feature_three_browse-bin_1&ds=v1%3AkenbXy%2FIUkzR4LEv2vjdkD20G5cm9r7NHFsdunAMaQ4"
    # print("Starting to scrape Amazon...")
    # df_4 = scrape_amazon(base_url_4, max_pages=75)
    # base_url_5 = "https://www.amazon.de/s?i=stripbooks&bbn=186606&rh=n%3A89265492031%2Cp_n_feature_three_browse-bin%3A15425222031&s=popularity-rank&dc&fs=true&qid=1736266441&rnid=4192708031&xpid=WLX86xT9ZvXkq&ref=sr_nr_p_n_feature_three_browse-bin_1&ds=v1%3A36rnLG3Ei8RTksIY7QKlSPMyXx5PQAjOfcX3VEGgbCg"
    # print("Starting to scrape Amazon...")
    # df_5 = scrape_amazon(base_url_5, max_pages=75)
    # base_url_6 = "https://www.amazon.de/s?k=romantisch&i=stripbooks&rh=n%3A142%2Cp_n_feature_three_browse-bin%3A15425222031%2Cp_n_binding_browse-bin%3A492558011%257C492559011%257C9330699031&dc&__mk_de_DE=%C3%85M%C3%85%C5%BD%C3%95%C3%91&qid=1736271116&rnid=492557011&xpid=eYSr8hxKCgLyD&ref=sr_pg_1"
    # print("Starting to scrape Amazon...")
    # df = scrape_amazon(base_url_6, max_pages=75)
    base_url_7 = "https://www.amazon.de/s?i=stripbooks&rh=n%3A186606%2Cn%3A5452843031%2Cn%3A5452877031%2Cn%3A16381996031%2Cp_n_binding_browse-bin%3A492558011%257C492559011%257C9330699031%2Cp_n_feature_three_browse-bin%3A15425222031%2Cp_n_availability%3A419126031&dc&qid=1736347790&rnid=5452877031&xpid=1-JK1OeR14wOO&ref=sr_pg_1"
    print("Starting to scrape Amazon...")
    df = scrape_amazon(base_url_7, max_pages=29)
    # base_url_8 = "https://www.amazon.de/s?i=stripbooks&rh=n%3A186606%2Cn%3A5452843031%2Cn%3A5452877031%2Cn%3A16381993031%2Cp_n_binding_browse-bin%3A492558011%257C492559011%257C9330699031%2Cp_n_feature_three_browse-bin%3A15425222031%2Cp_n_availability%3A419126031&dc&ds=v1%3APhiPu7%2FhIPRfXMggLPEzhKkxOO4wLP7x8WlORHxiz0A&qid=1736347955&rnid=5452877031&xpid=1-JK1OeR14wOO&ref=sr_nr_n_1"
    # print("Starting to scrape Amazon...")
    # df = scrape_amazon(base_url_8, max_pages=75)
    # base_url_9 = "https://www.amazon.de/s?i=stripbooks&rh=n%3A186606%2Cn%3A5452843031%2Cn%3A5452877031%2Cn%3A16381996031%2Cp_n_binding_browse-bin%3A492558011%257C492559011%257C9330699031%2Cp_n_feature_three_browse-bin%3A15425222031%2Cp_n_availability%3A419126031&dc&qid=1736347790&rnid=5452877031&xpid=1-JK1OeR14wOO&ref=sr_pg_1"
    # print("Starting to scrape Amazon...")
    # df = scrape_amazon(base_url_9, max_pages=31)
    
    print("\nSaving data to CSV...")
    # Change the Number of the CSV according to the number of the scrape
    df.to_csv('amazon_romantic_fantasy_main_7.csv', index=False, encoding='utf-8-sig')
    print("Done!")

