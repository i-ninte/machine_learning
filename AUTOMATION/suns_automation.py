from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import pandas as pd
from datetime import datetime 
import os
import sys


application_path= os.path.dirname(sys.executable)
now= datetime.now()

#DDMMYYYY
dmy= now.strftime("%d%m%Y")


website = "https://www.thesun.co.uk/sports/football/"
path = "/home/nintech/chromedriver-linux64/chromedriver-linux64/chromedriver"



#headless mode
options= Options()
options.headless= True
service = Service(executable_path=path)
driver = webdriver.Chrome(service=service, options= options)
driver.get(website)


containers= driver.find_elements(by"xpath", value='//div[@class="teaser_copy-container"]')

titles= []
subtitles=[]
links=[]



for container in containers:
    container.find_element(by="xpath", value='./a/h2').text
    container.find_element(by="xpath", value='./a/p').text
    link= container.find_element(by="xpath", value='./a).get_attribute("href")
    titles.append(title)
    subtitles.append(subtitle)
    links.append(link)






my_dict= {'title': titles, 'subtitle':subtitles, 'link': links}
df_headlines= pd.DataFrame(my_dict)
filename= f"headline{dmy}.csv"

final_path= os.path.join(application_path, filename)
#export to csv
df_headlines.to_csv(final_path)


driver.quit()



