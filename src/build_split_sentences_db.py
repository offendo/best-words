from tqdm import tqdm
import data
import os


OUTDIR = os.path.join('data', 'wiki_clean')
WIKI_PAGES = os.path.join('data', 'wiki-pages')

if __name__ == "__main__":
    for file in tqdm(os.listdir(WIKI_PAGES)):
        input_file = os.path.join(WIKI_PAGES, file)
        wiki = data.get_wiki(input_file)
        lines = wiki["lines"].apply(lambda l: "<SPLIT>".join(data.clean_article(l)))
        wiki["text"] = lines
        wiki = wiki.drop("lines", axis=1).reset_index()
        new_file = os.path.join(OUTDIR, file)
        wiki.to_json(new_file, orient="records", lines=True)
