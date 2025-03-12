from bs4 import BeautifulSoup
import json
import requests
def webscraper(url, output='webdata.json'):

    response = requests.get(url)
    htmlcontent = response.text
    #gets all the text from the website
    parse = BeautifulSoup(htmlcontent, 'html.parser') 
    #parses the the text using beautifulsoup

    
    Data = [p.get_text() for p in parse.find_all('p')]
    #has the parsed text extracted into data

    with open('Websiteinfo.json', 'w') as f:
        #website info is gonna be the name of the json file that has the info from the website

        json.dump(Data, f, indent = 4)
        #dumps all the data into a json file 

if __name__ == '__main__':
    #the url you want to use the webscraper on
    url = "https://www.acquisition.gov/far/part-1"
    webscraper(url)