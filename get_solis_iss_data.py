import requests
import urllib.request
import time
from bs4 import BeautifulSoup

# Set the URL you want to webscrape from
url = 'https://solis.nso.edu/pubkeep/ISS%201083.0%20(He%20I),%20level%202%20(spectra)%20_i22/'
# Connect to the URL
response = requests.get(url)

# Parse HTML and save to BeautifulSoup object¶
soup = BeautifulSoup(response.text, "html.parser")

#loop through all year/month folders
for i in range(5,len(soup.findAll('a'))): #'a' tags are for links
	one_a_tag = soup.findAll('a')[i]
	link = one_a_tag['href']
	#print(link)

	url2 = url+link
	response2 = requests.get(url2)
	soup2 = BeautifulSoup(response2.text, "html.parser")
	for j in range(5,len(soup2.findAll('a'))): 
		one_a_tag2 = soup2.findAll('a')[j]
		link2 = one_a_tag2['href']
		#print(link2)

		url3 = url2+link2
		response3 = requests.get(url3)
		soup3 = BeautifulSoup(response3.text, "html.parser")
		for k in range(5,len(soup3.findAll('a'))): 
			one_a_tag3 = soup3.findAll('a')[k]
			link3 = one_a_tag3['href']
			#print(link3)
			#file_type = "jpg"
			file_type = "fts.gz"
			
			if link3.endswith(file_type):
				download_url = url3 + link3
				#print(download_url)
				print(link3)
				urllib.request.urlretrieve(download_url, './'+link3)  # UNCOMMENT THIS line to download data
				time.sleep(1) #pause the code for a sec

