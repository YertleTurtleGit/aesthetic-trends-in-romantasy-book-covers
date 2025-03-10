import pandas as pd
import locale
import re


"""After this script run the save_imsges.py script to save the images."""

# Set locale to German for date parsing
locale.setlocale(locale.LC_ALL, 'de_DE')

def parse_date(date_str):
    try:
        # Remove unnecessary parts (e.g., "Illustriert,") and ensure there is a space between the day and month
        cleaned_date = date_str.split(",")[-1].strip()
        # Add a space after the dot if missing (e.g., "15.Januar" -> "15. Januar")
        cleaned_date = re.sub(r"(\d{1,2})\.(\w+)", r"\1. \2", cleaned_date)
        return pd.to_datetime(cleaned_date, format='%d. %B %Y', errors='raise')
    except ValueError:
        # Return NaT if the conversion fails
        return pd.NaT

# Read the CSV file
df = pd.read_csv("merged_amazon_romantic_fantasy.csv")

# Clean and convert the date column
df['Release Date'] = df['Release Date'].apply(parse_date)

# Print rows where the 'Release Date' could not be parsed
print(df[df['Release Date'].isna()])

# Group by title and select the earliest release date
idxmin = df.groupby(['Title'])['Release Date'].idxmin()

# Retrieve results based on the indices
result = df.loc[idxmin]

# Sort results by release date
result = result.sort_values('Release Date')

# Create a new ID column with consecutive numbering
result['ID'] = range(1, len(result) + 1)

# Save the result to a CSV file
result.to_csv("final_books.csv", index=False, encoding='utf-8-sig')
