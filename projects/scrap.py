# To scrape data from the Justdial website based on location and search string and save it to an Excel file
import requests
from bs4 import BeautifulSoup
import pandas as pd

def scrape_justdial(location, search_string):
    # Create an empty DataFrame to store the results
    df = pd.DataFrame(columns=["Name", "Phone", "Address", "Rating"])

    # Define the Justdial URL with location and search string
    url = f"https://www.justdial.com/{location}/{search_string}"

    # Send an HTTP GET request to the URL
    response = requests.get(url)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Parse the HTML content of the page
        soup = BeautifulSoup(response.text, 'html.parser')

        # Find all the listings on the page
        listings = soup.find_all('li', class_='cntanr')

        # Loop through each listing and extract relevant information
        for listing in listings:
            name = listing.find('span', class_='lng_cont_name').text.strip()
            phone = listing.find('p', class_='contact-info').text.strip()
            address = listing.find('span', class_='cont_fl_addr').text.strip()
            rating = listing.find('span', class_='green-box').text.strip()

            # Append the data to the DataFrame
            df = df.append({"Name": name, "Phone": phone, "Address": address, "Rating": rating}, ignore_index=True)

        # Save the data to an Excel file
        df.to_excel(f'justdial_{location}_{search_string}.xlsx', index=False)
        print(f'Data saved to justdial_{location}_{search_string}.xlsx')

    else:
        print("Failed to fetch data. Check your URL or internet connection.")

if __name__ == "__main__":
    location = input("Enter location (e.g., Mumbai): ")
    search_string = input("Enter search string (e.g., Restaurants): ")
    scrape_justdial(location, search_string)

