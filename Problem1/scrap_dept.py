
#file to scrapping content from each url and store in a text file

START_URL = "https://www.iitj.ac.in/m/Index/main-departments?lg=en"
OUTPUT_FILE = "iitj_departments.txt"

visited = set()
queue = [START_URL]
domain = urlparse(START_URL).netloc

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    text = re.sub(r"\s+", " ", text)

    # remove common junk
    patterns = [
        r"redirecttologinpage",
        r"all rights reserved",
        r"copyright.*",
        r"feedback",
        r"sitemap",
        r"important links.*"
    ]
    for p in patterns:
        text = re.sub(p, " ", text)

    return text.strip()

def get_text_and_links(url):
    try:
        res = requests.get(url, timeout=10)
        soup = BeautifulSoup(res.text, "lxml")

        for tag in soup(["script", "style", "noscript"]):
            tag.extract()

        text = soup.get_text(separator=" ")
        text = clean_text(text)

        links = []
        for a in soup.find_all("a", href=True):
            link = urljoin(url, a["href"])
            if urlparse(link).netloc == domain:
                links.append(link)

        return text, links
    except:
        return "", []

all_text = []

MAX_PAGES = 30

while queue and len(visited) < MAX_PAGES:
    url = queue.pop(0)

    if url in visited:
        continue

    visited.add(url)
    print("Scraping:", url)

    text, links = get_text_and_links(url)

    if text:
        all_text.append(f"# URL: {url}\n{text}\n")

    for link in links:
        if link not in visited:
            queue.append(link)

    time.sleep(0.5)

# Save final file
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write("\n\n".join(all_text))

print("\nSaved as:", OUTPUT_FILE)