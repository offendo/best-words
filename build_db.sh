#!/bin/bash
# Bash Menu Script to Build Skimmed DB, TFIDF, and Sentence Split DB

DBZIP=data/wiki_docs_skimmed.db.zip
TFIDFZIP=data/wiki_docs_skimmed-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz.zip

FEVER_WIKI_URL=https://s3-eu-west-1.amazonaws.com/fever.public/wiki-pages.zip

echo -e "\nPulling wiki-pages from source (FEVER) and building sentence split db...\n"
wget $FEVER_WIKI_URL -P data/
unzip -q data/wiki-pages.zip -d data/


echo -e "\nBuild split sentences db\n"
mkdir data/wiki_clean
python src/build_split_sentences_db.py
python src/utils/drqa_build_db.py data/wiki_clean/ data/wiki_docs_sentence_split.db


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
            echo -e "\nBuild source db"
            python3 src/utils/drqa_build_db.py data/wiki-pages/ data/wiki_docs.db
            echo -e "\nBuild skimmed db\n"
            python3 src/doc-retrieval/document_skimmer.py
            echo -e "\nBuild tfidf for skimmed db\n"
            python3 src/utils/drqa_build_tfidf.py data/wiki_docs_skimmed.db data/

            rm  data/wiki_docs.db
            break
            ;;
        "Quit")
            break
            ;;
        *) echo "invalid option $REPLY";;
    esac
done


echo -e "\nRemoving source files...\n"
rm -rf data/wiki_clean/
rm data/wiki-pages.zip