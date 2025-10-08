import os
import argparse
import pandas as pd
from time import time
from sqlalchemy import create_engine, text
from sklearn.metrics import classification_report

import logging
logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description='Connect to a PostgreSQL database.')
    parser.add_argument('--user', type=str, default='postgres', help='Database user name (default: postgres)')
    parser.add_argument('--password', type=str, default='', help='Database user password (default: none)')
    parser.add_argument('--database', type=str, default='bornsql', help='Name of the database (default: bornsql)')
    parser.add_argument('--host', type=str, default='localhost', help='Database host (default: localhost)')
    parser.add_argument('--port', type=int, default=5432, help='Database port (default: 5432)')
    return parser.parse_args()

def adult(engine, output):
    if not os.path.exists(output):
        os.makedirs(output)

    # Logger
    logger = logging.getLogger("adult")
    logger.addHandler(logging.FileHandler(os.path.join(output, "adult.log")))

    # Queries
    sql_encode = """
    SELECT madlib.encode_categorical_variables(
        'adult',
        'adult_encoded',
        'workclass, education, marital_status, occupation, relationship, race, sex, native_country',
        NULL,
        'id, target, split',
        NULL,
        NULL,
        FALSE,
        'array'
    );
    """
    sql_view_train = """
    CREATE VIEW adult_encoded_train AS
    SELECT id, "__encoded_variables__" AS features, CASE WHEN target = '>50K' THEN 1 ELSE 0 END AS target 
    FROM adult_encoded WHERE split = 'train';
    """
    sql_view_test = """
    CREATE VIEW adult_encoded_test AS
    SELECT id, "__encoded_variables__" AS features, CASE WHEN target = '>50K' THEN 1 ELSE 0 END AS target 
    FROM adult_encoded WHERE split = 'test';
    """
    sql_logregr_train = """
    SELECT madlib.logregr_train(
        'adult_encoded_train',
        'adult_logregr',
        'target',
        'features'
    );
    """
    sql_logregr_predict = """
    SELECT test.id, test.target AS y_true, madlib.logregr_predict(model.coef, features) AS y_pred
    FROM adult_encoded_test test, adult_logregr model;
    """
    sql_svm_train = """
    SELECT madlib.svm_classification(
        'adult_encoded_train',
        'adult_svm',
        'target',
        'features'
    );
    """
    sql_svm_predict = """
    SELECT madlib.svm_predict(
        'adult_svm', 
        'adult_encoded_test',
        'id', 
        'adult_svm_pred'
    );
    """
    sql_dt_train = """
    SELECT madlib.tree_train(
        'adult_encoded_train',
        'adult_dt',
        'id',
        'target',
        'features'
    );
    """
    sql_dt_predict = """
    SELECT madlib.tree_predict(
        'adult_dt', 
        'adult_encoded_test',
        'adult_dt_pred'
    );
    """

    # Transform, train, and predict
    with engine.connect() as con:
        con.execute(text("DROP VIEW IF EXISTS adult_encoded_train"))
        con.execute(text("DROP VIEW IF EXISTS adult_encoded_test"))        
        con.execute(text("DROP TABLE IF EXISTS adult_encoded"))
        con.execute(text("DROP TABLE IF EXISTS adult_encoded_dictionary"))        
        tic = time()
        con.execute(text(sql_encode))
        con.execute(text(sql_view_train))
        con.execute(text(sql_view_test))
        toc = time()
        logger.info(text(f"madlib sql_encode time (s): {toc-tic}"))

        con.execute(text("DROP TABLE IF EXISTS adult_logregr"))
        con.execute(text("DROP TABLE IF EXISTS adult_logregr_summary"))
        tic = time()
        con.execute(text(sql_logregr_train))
        toc = time()
        logger.info(text(f"madlib sql_logregr_train time (s): {toc-tic}"))

        tic = time()
        pred = pd.read_sql(sql_logregr_predict, con=con)
        toc = time()
        logger.info(f"madlib sql_logregr_predict time (s): {toc-tic}")
        logger.info(f"Report logregr: \n{classification_report(y_true=pred.y_true, y_pred=pred.y_pred)}")

        con.execute(text("DROP TABLE IF EXISTS adult_svm"))
        con.execute(text("DROP TABLE IF EXISTS adult_svm_summary"))
        tic = time()
        con.execute(text(sql_svm_train))
        toc = time()
        logger.info(text(f"madlib sql_svm_train time (s): {toc-tic}"))

        con.execute(text("DROP TABLE IF EXISTS adult_svm_pred"))
        tic = time()
        con.execute(text(sql_svm_predict))
        df = pd.read_sql("""
            SELECT adult_encoded_test.target AS y_true, adult_svm_pred.prediction AS y_pred 
            FROM adult_svm_pred JOIN adult_encoded_test USING (id);
        """, con=con)
        toc = time()
        logger.info(f"madlib sql_svm_predict time (s): {toc-tic}")
        logger.info(f"Report svm: \n{classification_report(y_true=df.y_true, y_pred=df.y_pred)}")

        con.execute(text("DROP TABLE IF EXISTS adult_dt"))
        con.execute(text("DROP TABLE IF EXISTS adult_dt_summary"))
        tic = time()
        con.execute(text(sql_dt_train))
        toc = time()
        logger.info(text(f"madlib sql_dt_train time (s): {toc-tic}"))

        con.execute(text("DROP TABLE IF EXISTS adult_dt_pred"))
        tic = time()
        con.execute(text(sql_dt_predict))
        df = pd.read_sql("""
            SELECT adult_encoded_test.target AS y_true, adult_dt_pred.estimated_target AS y_pred 
            FROM adult_dt_pred JOIN adult_encoded_test USING (id);
        """, con=con)
        toc = time()
        logger.info(f"madlib sql_dt_predict time (s): {toc-tic}")
        logger.info(f"Report dt: \n{classification_report(y_true=df.y_true, y_pred=df.y_pred)}")

        con.commit()

def rlcp(engine, output):
    if not os.path.exists(output):
        os.makedirs(output)

    # Logger
    logger = logging.getLogger("rlcp")
    logger.addHandler(logging.FileHandler(os.path.join(output, "rlcp.log")))

    # Queries
    sql_encode = """
    SELECT madlib.encode_categorical_variables(
        'rlcp',
        'rlcp_encoded',
        'cmp_fname_c1=1, cmp_fname_c2=1, cmp_lname_c1=1, cmp_lname_c2=1, cmp_sex, cmp_bd::INTEGER, cmp_bm::INTEGER, cmp_by::INTEGER, cmp_plz::INTEGER',
        NULL,
        'id, is_match',
        NULL,
        NULL,
        FALSE,
        'array'
    );
    """
    sql_view_train = """
    CREATE VIEW rlcp_encoded_train AS
    SELECT id, "__encoded_variables__" AS features, is_match
    FROM rlcp_encoded WHERE id <= 4600000;
    """
    sql_view_test = """
    CREATE VIEW rlcp_encoded_test AS
    SELECT id, "__encoded_variables__" AS features, is_match
    FROM rlcp_encoded WHERE id > 4600000;
    """
    sql_logregr_train = """
    SELECT madlib.logregr_train(
        'rlcp_encoded_train',
        'rlcp_logregr',
        'is_match',
        'features'
    );
    """
    sql_logregr_predict = """
    SELECT test.id, test.is_match AS y_true, madlib.logregr_predict(model.coef, features) AS y_pred
    FROM rlcp_encoded_test test, rlcp_logregr model;
    """
    sql_svm_train = """
    SELECT madlib.svm_classification(
        'rlcp_encoded_train',
        'rlcp_svm',
        'is_match',
        'features'
    );
    """
    sql_svm_predict = """
    SELECT madlib.svm_predict(
        'rlcp_svm', 
        'rlcp_encoded_test',
        'id', 
        'rlcp_svm_pred'
    );
    """
    sql_dt_train = """
    SELECT madlib.tree_train(
        'rlcp_encoded_train',
        'rlcp_dt',
        'id',
        'is_match',
        'features'
    );
    """
    sql_dt_predict = """
    SELECT madlib.tree_predict(
        'rlcp_dt', 
        'rlcp_encoded_test',
        'rlcp_dt_pred'
    );
    """

    # Transform, train, and predict
    with engine.connect() as con:
        con.execute(text("DROP VIEW IF EXISTS rlcp_encoded_train"))
        con.execute(text("DROP VIEW IF EXISTS rlcp_encoded_test"))        
        con.execute(text("DROP TABLE IF EXISTS rlcp_encoded"))
        con.execute(text("DROP TABLE IF EXISTS rlcp_encoded_dictionary"))        
        tic = time()
        con.execute(text(sql_encode))
        con.execute(text(sql_view_train))
        con.execute(text(sql_view_test))
        toc = time()
        logger.info(text(f"madlib sql_encode time (s): {toc-tic}"))

        con.execute(text("DROP TABLE IF EXISTS rlcp_logregr"))
        con.execute(text("DROP TABLE IF EXISTS rlcp_logregr_summary"))
        tic = time()
        con.execute(text(sql_logregr_train))
        toc = time()
        logger.info(text(f"madlib sql_logregr_train time (s): {toc-tic}"))

        tic = time()
        pred = pd.read_sql(sql_logregr_predict, con=con)
        toc = time()
        logger.info(f"madlib sql_logregr_predict time (s): {toc-tic}")
        logger.info(f"Report logregr: \n{classification_report(y_true=pred.y_true, y_pred=pred.y_pred)}")

        con.execute(text("DROP TABLE IF EXISTS rlcp_svm"))
        con.execute(text("DROP TABLE IF EXISTS rlcp_svm_summary"))
        tic = time()
        con.execute(text(sql_svm_train))
        toc = time()
        logger.info(text(f"madlib sql_svm_train time (s): {toc-tic}"))

        con.execute(text("DROP TABLE IF EXISTS rlcp_svm_pred"))
        tic = time()
        con.execute(text(sql_svm_predict))
        df = pd.read_sql("""
            SELECT rlcp_encoded_test.is_match AS y_true, rlcp_svm_pred.prediction AS y_pred 
            FROM rlcp_svm_pred JOIN rlcp_encoded_test USING (id);
        """, con=con)
        toc = time()
        logger.info(f"madlib sql_svm_predict time (s): {toc-tic}")
        logger.info(f"Report svm: \n{classification_report(y_true=df.y_true, y_pred=df.y_pred)}")

        con.execute(text("DROP TABLE IF EXISTS rlcp_dt"))
        con.execute(text("DROP TABLE IF EXISTS rlcp_dt_summary"))
        tic = time()
        con.execute(text(sql_dt_train))
        toc = time()
        logger.info(text(f"madlib sql_dt_train time (s): {toc-tic}"))

        con.execute(text("DROP TABLE IF EXISTS rlcp_dt_pred"))
        tic = time()
        con.execute(text(sql_dt_predict))
        df = pd.read_sql("""
            SELECT rlcp_encoded_test.is_match AS y_true, rlcp_dt_pred.estimated_is_match AS y_pred 
            FROM rlcp_dt_pred JOIN rlcp_encoded_test USING (id);
        """, con=con)
        toc = time()
        logger.info(f"madlib sql_dt_predict time (s): {toc-tic}")
        logger.info(f"Report dt: \n{classification_report(y_true=df.y_true, y_pred=df.y_pred)}")
        
        con.commit()

if __name__ == "__main__":
    args = parse_args()

    # Create an engine to the PostgreSQL database
    engine = create_engine(
        url=f"postgresql+psycopg2://{args.user}:{args.password}@{args.host}:{args.port}/{args.database}",
        executemany_mode='values_plus_batch',
        insertmanyvalues_page_size=1000,
        executemany_batch_page_size=1000
    )

    # Adult dataset
    print("Running adult...")
    output = 'output/adult'
    adult(engine, output)

    # RLCP dataset
    print("Running rlcp...")
    output = 'output/rlcp'
    rlcp(engine, output)
