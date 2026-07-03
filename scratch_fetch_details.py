import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET

ids = ["2307.07863", "2408.01244", "1905.00336"]
url = f"http://export.arxiv.org/api/query?id_list={','.join(ids)}"

try:
    with urllib.request.urlopen(url) as response:
        xml_data = response.read()
        root = ET.fromstring(xml_data)
        
        ns = {
            'atom': 'http://www.w3.org/2005/Atom'
        }
        
        entries = root.findall('atom:entry', ns)
        with open("results/detailed_literature_results.txt", "w", encoding="utf-8") as f:
            for i, entry in enumerate(entries):
                title = entry.find('atom:title', ns).text.strip().replace('\n', ' ')
                published = entry.find('atom:published', ns).text
                summary = entry.find('atom:summary', ns).text.strip().replace('\n', ' ')
                id_url = entry.find('atom:id', ns).text
                authors = ", ".join([author.find('atom:name', ns).text for author in entry.findall('atom:author', ns)])
                
                output = f"Paper [{i+1}] ID: {id_url}\nTitle: {title}\nYear: {published}\nAuthors: {authors}\nAbstract: {summary}\n\n"
                f.write(output)
        print("Detailed literature metadata fetched successfully!")
except Exception as e:
    print(f"Error fetching details: {e}")
