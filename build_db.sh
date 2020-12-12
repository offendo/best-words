#!/bin/bash
# Bash Menu Script Example

DBZIP=data/wiki_docs_skimmed.db.zip
TFIDFZIP=data/wiki_docs_skimmed-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz.zip

FEVER_WIKI_URL=https://s3-eu-west-1.amazonaws.com/fever.public/wiki-pages.zip

PS3='Please enter your choice: '
options=("Build skimmed db from .zip" "Build skimmed db from source (FEVER)" "Quit")
select opt in "${options[@]}"
do
    case $opt in
        "Build skimmed db from .zip")
            echo -e "\nunzipping wiki_docs_skimmed.db.zip ...\n"
            unzip -q $DBZIP -d data/
            echo -e "\nunzipping wiki_docs_skimmed tf-idf ...\n"
            unzip -q $TFIDFZIP -d data/
            echo -e "Done."
            break
            ;;
        "Build skimmed db from source (FEVER)")
            echo -e "\npulling from source (FEVER)\n"
            # wget $FEVER_WIKI_URL -P data/
            unzip -q data/wiki-pages.zip -d data/
            echo -e "\nBuild source db"
            python3 src/utils/drqa_build_db.py data/wiki-pages/ data/wiki_docs.db
            echo -e "\nBuild skimmed db\n"
            python3 src/doc-retrieval/document_skimmer.py
            echo -e "\nBuild tfidf for skimmed db\n"
            python3 src/utils/drqa_build_tfidf.py data/wiki_docs_skimmed.db data/
            echo -e "\nRemoving source files...\n"
            rm -rf wiki-pages/
            rm wiki-pages.zip wiki_docs.db
            break
            ;;
        "Quit")
            break
            ;;
        *) echo "invalid option $REPLY";;
    esac
done