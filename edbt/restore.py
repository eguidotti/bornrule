import os
import argparse
import pandas as pd
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, BigInteger, Text, String


def parse_args():
    parser = argparse.ArgumentParser(description='Connect to a database.')
    parser.add_argument('--user', type=str, default='postgres', help='Database user name (default: postgres)')
    parser.add_argument('--password', type=str, default='', help='Database user password (default: none)')
    parser.add_argument('--database', type=str, default='bornsql', help='Name of the database (default: bornsql)')
    parser.add_argument('--host', type=str, default='localhost', help='Database host (default: localhost)')
    parser.add_argument('--port', type=int, default=5432, help='Database port (default: 5432)')
    parser.add_argument('--dbms', type=str, default='postgresql', choices=['postgresql', 'mysql', 'sqlite'], 
                        help='Database management system (default: postgresql)')
    return parser.parse_args()

def restore_rlcp(engine):
    with engine.connect() as con:
        if not engine.dialect.has_table(con, 'rlcp'):
            blocks = pd.concat([pd.read_csv(f'data/rlcp/block_{i+1}.csv') for i in range(10)], ignore_index=True)
            blocks.to_sql('rlcp', con, index_label='id', if_exists='append')
            con.commit()

def restore_scopus(args):
    if args.dbms == 'postgresql':
        dump = "data/scopus/scopus.dump"
        options = "--no-owner --exit-on-error"
        pwd = f"PGPASSWORD='{args.password}'"
        os.system(f"{pwd} pg_restore -h {args.host} -p {args.port} -U {args.user} -d {args.database} {options} {dump}")
    else:
        MyString = String(512, collation='utf8mb4_bin') if args.dbms == 'mysql' else String(512)
        metadata = MetaData()
        publication = Table(
            'publication', metadata,
            Column('id', BigInteger, primary_key=True),
            Column('pubname', MyString),
            Column('asjc', Integer),
            Column('abstract', Text)
        )
        pub_author = Table(
            'pub_author', metadata,
            Column('pubid', BigInteger, primary_key=True),
            Column('authid', BigInteger, primary_key=True)
        )
        pub_keyword = Table(
            'pub_keyword', metadata,
            Column('pubid', BigInteger, primary_key=True),
            Column('keyword', MyString, primary_key=True)
        )
        metadata.create_all(engine)
        with engine.connect() as con:
            pd.read_csv('data/scopus/publication.csv.gz').to_sql('publication', con, if_exists='append', index=False)
            pd.read_csv('data/scopus/pub_author.csv.gz').to_sql('pub_author', con, if_exists='append', index=False)
            pd.read_csv('data/scopus/pub_keyword.csv.gz').fillna('').to_sql('pub_keyword', con, if_exists='append', index=False)
            con.commit()

def restore_adult(engine):
    names = [
        'age',
        'workclass',
        'fnlwgt',
        'education',
        'education_num',
        'marital_status',
        'occupation',
        'relationship',
        'race',
        'sex',
        'capital_gain',
        'capital_loss',
        'hours_per_week',
        'native_country',
        'target'
    ]
    
    train = pd.read_csv('data/adult/adult.data', header=None, names=names, sep=', ', engine='python')
    test = pd.read_csv('data/adult/adult.test', header=None, names=names, sep=', ', engine='python', skiprows=1)
    test.target = test.target.str.replace('.', '')
    train['split'] = 'train'
    test['split'] = 'test'
    
    with engine.connect() as con:
        if not engine.dialect.has_table(con, 'adult'):
            pd.concat([train, test]).reset_index().to_sql('adult', con, index_label='id')
            con.commit()


if __name__ == "__main__":
    args = parse_args()

    # Create an engine to the database
    if args.dbms == 'postgresql':
        engine = create_engine(
            url=f"postgresql+psycopg2://{args.user}:{args.password}@{args.host}:{args.port}/{args.database}",
            executemany_mode='values_plus_batch',
            insertmanyvalues_page_size=1000,
            executemany_batch_page_size=1000
        )
    elif args.dbms == 'mysql':
        engine = create_engine(
            url=f"mysql+pymysql://{args.user}:{args.password}@{args.host}:{args.port}/{args.database}",
            insertmanyvalues_page_size=1000
        )
    elif args.dbms == 'sqlite':
        engine = create_engine(
            url=f"sqlite:///{args.database}.db"
        )
    else:
        raise ValueError(f"Unsupported DBMS: {args.dbms}")

    # Restore datasets
    print("Restoring scopus...")
    restore_scopus(args)
    if args.dbms == 'postgresql':        
        print("Restoring adult...")
        restore_adult(engine)
        print("Restoring rlcp...")
        restore_rlcp(engine)
