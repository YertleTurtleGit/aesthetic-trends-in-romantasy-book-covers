import os
import requests
import pandas as pd

""" After this script run blip_description.py to get the book descriptions."""

# Read the CSV file into a DataFrame
df = pd.read_csv('final_books.csv')

def save_images(df, folder='images'):
    """ Save images from URLs to a local folder """
    
    # Check if the specified folder exists -> if not, create it
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Add a new column 'Local_Image_Path' to store the local paths of the saved images
    df['Local_Image_Path'] = None
    
    # Loop through each row in the DataFrame
    for index, row in df.iterrows():
        # Skip rows where the 'Image_URL' is missing (NaN)
        if pd.isna(row['Image_URL']):
            continue
        
        # Extract the image URL and generate a local file name based on the 'ID' column
        image_url = row['Image_URL']
        image_name = f"{row['ID']}.jpg"
        image_path = os.path.join(folder, image_name)  # Define the local path where the image will be saved
        
        try:
            # Download the image from the URL
            response = requests.get(image_url, stream=True)
            
            # Checks if the request was successful (status code 200)
            if response.status_code == 200:
                # Saves the image in chunks to the specified local path
                with open(image_path, 'wb') as f:
                    for chunk in response.iter_content(1024):  # Write the image in 1KB chunks
                        f.write(chunk)
                # Update the DataFrame with the local path of the saved image
                df.at[index, 'Local_Image_Path'] = image_path
                print(f"Saved image: {image_name}")  
            
        except Exception as e:
            print(f"Error saving image {image_url}: {e}")
    
    return df  # Return the updated DataFrame with local image paths

if __name__ == "__main__":
    # Start the image download process
    print("\nDownloading images...")
    df = save_images(df) 
    print("We did it :)")  