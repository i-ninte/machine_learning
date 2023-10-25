# to scrap the images and articles from a website. based on specific keywords 
import requests
from bs4 import BeautifulSoup
import os
import urllib

# Define the URL of the website to scrape
url = 'https://example.com'  # Replace with the URL of the target website

# Define the keywords you want to search for
keywords = ['keyword1', 'keyword2', 'keyword3']  # Add your desired keywords

# Create a directory to store downloaded images
output_directory = 'downloaded_images'
os.makedirs(output_directory, exist_ok=True)

# Send an HTTP request to the website
response = requests.get(url)

# Check if the request was successful
if response.status_code == 200:
    # Parse the HTML content of the webpage
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Find all the articles on the webpage
    articles = soup.find_all('article')
    
    # Loop through the articles to extract images, headings, and descriptions
    for article in articles:
        heading = article.find('h2').text.strip()
        description = article.find('p').text.strip()
        images = article.find_all('img')
        
        # Check if any of the keywords appear in the heading or description
        for keyword in keywords:
            if keyword in heading or keyword in description:
                # Download images related to the keyword
                for img in images:
                    img_url = img['src']
                    img_name = img_url.split('/')[-1]
                    img_path = os.path.join(output_directory, img_name)
                    urllib.request.urlretrieve(img_url, img_path)
                    
                # Print the heading and description
                print("Heading:", heading)
                print("Description:", description)
                print("Images related to keyword:", keyword)
                print()
else:
    print("Failed to retrieve the webpage. Status code:", response.status_code)
