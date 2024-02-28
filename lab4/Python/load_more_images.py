import requests
import os
from bs4 import BeautifulSoup
from celebrities import celebrities
#     return False    
'''if internet_connection():
    print("The Internet is connected.")
else:
    print("The Internet is not connected.")

internet_connection'''

def get_thumbnails_google(search_term, count=50):
    thumbnails = []
    try:
        url = rf'https://www.google.com/search?q={search_term}&tbm=isch'
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        for raw_img in soup.find_all('img'):
            link = raw_img.get('src')
            if link and link.startswith("https://"):
                thumbnails.append(link)
                if len(thumbnails) >= count:
                    return thumbnails
    except Exception as e:
        print(f"Error fetching images from Google: {e}")
    return thumbnails

def get_thumbnails_bing(search_term, count=50):
    thumbnails = []
    try:
        url = rf'https://www.bing.com/images/search?q={search_term}'
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        for img in soup.find_all('img'):
            link = img.get('src')
            if link and link.startswith("https://"):
                thumbnails.append(link)
                if len(thumbnails) >= count:
                    return thumbnails
    except Exception as e:
        print(f"Error fetching images from Bing: {e}")
    return thumbnails

def get_thumbnails_yahoo(search_term, count=200):
    thumbnails = []
    try:
        url = rf'https://images.search.yahoo.com/search/images?p={search_term}'
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        for img in soup.find_all('img'):
            link = img.get('src')
            if link and link.startswith("https://"):
                thumbnails.append(link)
                if len(thumbnails) >= count:
                    return thumbnails
    except Exception as e:
        print(f"Error fetching images from Yahoo: {e}")
    return thumbnails

def save_thumbnails(search_term, thumbnails):
    try:
        folder_name = f"more_photos/{search_term}"
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        for i, thumbnail in enumerate(thumbnails):
            filename = os.path.join(folder_name, f"{search_term}_{i}.jpg")
            with open(filename, 'wb') as f:
                f.write(requests.get(thumbnail).content)
            print(f"Image {i + 1} saved as {filename}")
    except Exception as e:
        print(f"Error saving images: {e}")

for name in celebrities:
    thumbnails_google = get_thumbnails_google(name)
    # thumbnails_bing = get_thumbnails_bing(name)
    thumbnails_yahoo = get_thumbnails_yahoo(name)
    all_thumbnails = thumbnails_google + thumbnails_yahoo # thumbnails_bing
    save_thumbnails(name, all_thumbnails)

'''thumbnails_google = get_thumbnails_google('flowers')
thumbnails_bing = get_thumbnails_bing('flowers')
thumbnails_yahoo = get_thumbnails_yahoo('flowers')
all_thumbnails = thumbnails_google + thumbnails_bing + thumbnails_yahoo
save_thumbnails('flowers', all_thumbnails)

print(len(thumbnails_google))
print(len(thumbnails_bing))
print(len(thumbnails_yahoo))'''
