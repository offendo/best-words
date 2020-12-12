import json
import itertools
import sqlite3
import os

data_dir = os.path.join('data')
db_file = os.path.join(data_dir, 'wiki_docs_skimmed.db')
whole_db = os.path.join(data_dir, 'wiki_docs.db')
train_jsonl = os.path.join(data_dir, 'train.jsonl')


def db_create_table(db_filepath):
    create_table = '''
        create table if not exists documents (
            id text primary key,
            text text not null
        )
    '''
    with sqlite3.connect(db_filepath) as conn:
        cur = conn.cursor()
        cur.execute(create_table)


def db_insert_rows(db_filepath, rows):
    insert = '''
        insert or ignore into documents
        values {};
    '''
    tupl_str_art = list(map(lambda art: "('{}', '{}')".format(art[0].replace("'", "''"), art[1].replace("'", "''")), rows))

    insert_values = insert.format(', '.join(tupl_str_art))

    with sqlite3.connect(db_filepath) as conn:
        cur = conn.cursor()
        cur.execute(insert_values)
        cur.execute('select count(1) from documents')
        count = cur.fetchone()
    return count


if __name__ == "__main__":
    n_extra = int (input('Enter # extra documents per claim evidence (def: 9): ') or '9')

    all_articles = set()
    with open(train_jsonl, 'r') as fp:
        claims = []
        for jsonl in fp.readlines():
            jsoned = json.loads(jsonl)
            for evidence in itertools.chain(*jsoned['evidence']):
                if evidence[2] is not None:
                    all_articles.add(evidence[2])
            claims.append(jsoned)

    # join article ids to get selected from db
    articles_joined = ', '.join(map(lambda art: "'{}'".format(art.replace("'", "''")), all_articles))
    sql_articles = 'select * from documents where id in ({})'.format(articles_joined)

    # get all articles
    with sqlite3.connect(whole_db) as conn:
        cur = conn.cursor()
        cur.execute(sql_articles)
        article_rows = cur.fetchall()

    db_create_table(db_file)
    db_insert_rows(db_file, article_rows)

    # add extra documents from articles
    # CANNOT SEED. will be random every time
    extra_docs = n_extra * len(all_articles)
    with sqlite3.connect(whole_db) as conn:
        cur = conn.cursor()
        cur.execute('SELECT * FROM documents ORDER BY RANDOM() LIMIT {};'.format(extra_docs))
        unknown_arts = cur.fetchall()

    total_count = db_insert_rows(db_file, unknown_arts)
    print('wiki_docs_skimmed.db contains {} documents.'.format(total_count[0]))
