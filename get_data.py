""" Download Pushshift Data """

import os
from urllib import request as req
import re
import pycurl
import hashlib
# Define values

URLS=["https://files.pushshift.io/reddit/comments/","https://files.pushshift.io/reddit/submissions"]
BZ2_LINK_RE_PATTERN = r"<a\s.*href=[\"'](\S+)[\"'][^>]*>\S*.bz2<\/a>"
SHA256_LINK_RE_PATTERN = r"<a\s.*href=[\"'](\S+)[\"'][^>]*>sha256\S*<\/a>"
OUTPUT_DIR = "reddit_submissions"
# Define functions
def main():
    """The main entrypoint."""

    for BASE_URL in URLS:
        submissions_page=req.urlopen(BASE_URL).read().decode("utf-8")


        # Get BZ2 Links
        raw_links = re.findall(BZ2_LINK_RE_PATTERN,submissions_page)
        filtered_links = [link[2:] for link in raw_links if link.startswith("./")]
        individual_links = list(set(filtered_links))

        # Download files
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
        else:
            # get first match and remove the ./ from the start of the link
            sha256_link = re.findall(SHA256_LINK_RE_PATTERN,submissions_page)[0][2:]
            hash_file=(req.urlopen("%s/%s"%(BASE_URL,sha256_link)).read().decode("utf-8"))

            hash_file_pairs=({entry.split("  ")[1]:entry.split("  ")[0] for entry in hash_file.split("\n") if len(entry.split("  "))>1})
            for file in hash_file_pairs.keys():
                file_path=os.path.join(OUTPUT_DIR,file)
                if(os.path.exists(file_path) and hashlib.sha256(open(file_path,'rb').read()).hexdigest()!=hash_file_pairs[file.split("/")[-1]]):
                    print("File is corrput, deleting %s"%file_path)
                    os.remove(file_path)


        curl = pycurl.Curl()
        for link in sorted(individual_links):
            url = BASE_URL + "/" + link
            file_path=os.path.join(OUTPUT_DIR, link)

            if not os.path.exists(file_path):
                with open(file_path, "wb") as file:
                    curl.setopt(curl.URL, url)
                    curl.setopt(curl.WRITEDATA, file)
                    curl.perform()
                print("Downloaded", link)
        curl.close()

# Execute main function
if __name__ == "__main__":
    main()
