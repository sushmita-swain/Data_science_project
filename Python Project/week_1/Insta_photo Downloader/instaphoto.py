from os import close
import requests
from bs4 import *
from selenium import webdriver
import urllib

header = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.111 Safari/537.36"}
url = "https://www.instagram.com/python.hub/?hl=en"
response = requests.get(url, headers = header)

# print(response.text)

soup = BeautifulSoup(response.text, "html.parser")
# print(soup)

DRIVER_PATH = "/home/sushmita/Python Project/Insta_photo Downloader/chromedriver"
driver = webdriver.Chrome(executable_path= DRIVER_PATH)

driver.get("https://www.instagram.com/python.hub/?hl=en")
driver.page_source

soup = BeautifulSoup(driver.page_source, "html.parser")
# print(soup)

# a = soup.find_all('a', href = True)[0]["href"]
# print(a)

links = []

for i in soup.find_all("a", href= True):
    if i["href"].startswith("/p"):
        print("Link Found : https://www.instagram.com/{0}".format(i["href"]))
        links.append("https://www.instagram.com/" + i["href"])


def download_image(url , destination = '/home/sushmita/Python Project/Insta_photo Downloader/images'):
    resource = urllib.requests.urlopen(url)
    filename = destination + url[-8:] + '.jpg'
    output = open(filename, "wb")
    output.write(resource.read())
    output,close

for i,j in enumerate(links):
    driver_i = webdriver.Chrome(executable_path= DRIVER_PATH)
    driver_i.get("https://www.instagram.com//p/CG5AnjEgVPp/")
    soup_i = BeautifulSoup(driver_i.page_source, "html.parser")
    image_link = soup_i.find_all('div', {'class': 'eLAPa RzuR0'})[0].find_all('img')[0]['src']
    download_image(image_link)
    driver_i.quit()
   





