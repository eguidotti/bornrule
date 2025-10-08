import os
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sqlalchemy import create_engine, text, Integer, String, Boolean
from bornrule.sql import BornClassifierSQL
from sklearn.metrics import classification_report
from time import time
from dataset import Dataset

import logging
logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description='Connect to a PostgreSQL database.')
    parser.add_argument('--user', type=str, default='postgres', help='Database user name (default: postgres)')
    parser.add_argument('--password', type=str, default='', help='Database user password (default: none)')
    parser.add_argument('--database', type=str, default='bornsql', help='Name of the database (default: bornsql)')
    parser.add_argument('--host', type=str, default='localhost', help='Database host (default: localhost)')
    parser.add_argument('--port', type=int, default=5432, help='Database port (default: 5432)')
    parser.add_argument('--dbms', type=str, choices=['postgresql', 'mysql', 'sqlite'], default='postgresql', help='DBMS (default: postgresql)')
    return parser.parse_args()

def adult(engine, output):
    if not os.path.exists(output):
        os.makedirs(output)
    
    # Logger
    logger = logging.getLogger("adult")
    logger.addHandler(logging.FileHandler(os.path.join(output, "adult.log")))
    
    # Define configs for class and featre
    configs = {
        'class': f"SELECT id as item, target AS class, 1.0 AS weight FROM adult",    
        'features': [
            f"SELECT id as item, 'workclass:'||workclass as feature, 1.0 as weight FROM adult",
            f"SELECT id as item, 'education:'||education as feature, 1.0 as weight FROM adult",
            f"SELECT id as item, 'marital_status:'||marital_status as feature, 1.0 as weight FROM adult",
            f"SELECT id as item, 'occupation:'||occupation as feature, 1.0 as weight FROM adult",
            f"SELECT id as item, 'relationship:'||relationship as feature, 1.0 as weight FROM adult",                
            f"SELECT id as item, 'race:'||race as feature, 1.0 as weight FROM adult",                
            f"SELECT id as item, 'sex:'||sex as feature, 1.0 as weight FROM adult",                
            f"SELECT id as item, 'native_country:'||native_country as feature, 1.0 as weight FROM adult",                
        ]   
    }
    
    # Train
    bc = BornClassifierSQL(id="adult", engine=engine, configs=configs, type_class=MyString, type_feature=MyString)
    bc.undeploy(deep=True)
    tic = time()
    bc.fit("SELECT id as item FROM adult WHERE split = 'train'")
    toc = time()
    logger.info(f"bornsql training time (s): {toc-tic}")

    # Deploy
    tic = time()
    bc.deploy()
    toc = time()
    logger.info(f"bornsql deploy time (s): {toc-tic}")

    # Test
    tic = time()
    y_pred = bc.predict("SELECT id as item FROM adult WHERE split = 'test'")
    toc = time()
    logger.info(f"bornsql inference time (s): {toc-tic}")

    # Evaluate
    with engine.connect() as con:
        y_true = pd.read_sql("SELECT id, target FROM adult WHERE split = 'test'", con)
        y_true = pd.Series(y_true.target.values, index=y_true.id.values)
    y_true, y_pred = y_true.values == '>50K', y_pred[y_true.index].values == '>50K'    
    logger.info(f"Report bornsql: \n{classification_report(y_true=y_true, y_pred=y_pred)}")

    # Global explanation
    ex_global = bc.explain()
    ex_global = ex_global.melt(ignore_index=False).reset_index().sort_values(by='value', ascending=False)
    ex_global.columns = ['feature', 'class', 'weight']
    ex_global = ex_global.groupby('class').head(100)
    ex_global = ex_global.reset_index(drop=True).sort_values(by=['class', 'weight'], ascending=False)
    pd.DataFrame(ex_global).to_csv(os.path.join(output, "explain_global.csv"), index=False)

def rlcp(engine, output):
    if not os.path.exists(output):
        os.makedirs(output)
    
    # Logger
    logger = logging.getLogger("rlcp")
    logger.addHandler(logging.FileHandler(os.path.join(output, "rlcp.log")))

    # Define configs for class and features
    configs = {
        'class': "SELECT id as item, is_match AS class, 1.0 AS weight FROM rlcp",    
        'features': [
            "SELECT id as item, CASE WHEN cmp_fname_c1 = 1 THEN 'cmp_fname_c1:1' ELSE 'cmp_fname_c1:0' END as feature, 1.0 AS weight FROM rlcp",    
            "SELECT id as item, CASE WHEN cmp_fname_c2 = 1 THEN 'cmp_fname_c2:1' ELSE 'cmp_fname_c2:0' END as feature, 1.0 AS weight FROM rlcp",    
            "SELECT id as item, CASE WHEN cmp_lname_c1 = 1 THEN 'cmp_lname_c1:1' ELSE 'cmp_lname_c1:0' END as feature, 1.0 AS weight FROM rlcp",    
            "SELECT id as item, CASE WHEN cmp_lname_c2 = 1 THEN 'cmp_lname_c2:1' ELSE 'cmp_lname_c2:0' END as feature, 1.0 AS weight FROM rlcp",    
            "SELECT id as item, 'cmp_sex:'||cmp_sex as feature, 1.0 AS weight FROM rlcp",    
            "SELECT id as item, 'cmp_bd:'||cmp_bd as feature, 1.0 AS weight FROM rlcp",    
            "SELECT id as item, 'cmp_bm:'||cmp_bm as feature, 1.0 AS weight FROM rlcp",    
            "SELECT id as item, 'cmp_by:'||cmp_by as feature, 1.0 AS weight FROM rlcp",    
            "SELECT id as item, 'cmp_plz:'||cmp_plz as feature, 1.0 AS weight FROM rlcp",    
        ]   
    }
    
    # Train
    bc = BornClassifierSQL(id="rlcp", engine=engine, configs=configs, type_class=Boolean, type_feature=MyString)
    bc.undeploy(deep=True)
    tic = time()
    bc.fit("SELECT id as item FROM rlcp WHERE id <= 4600000")
    toc = time()
    logger.info(f"bornsql training time (s): {toc-tic}")

    # Deploy
    tic = time()
    bc.deploy()
    toc = time()
    logger.info(f"bornsql deploy time (s): {toc-tic}")

    # Test
    tic = time()
    y_pred = bc.predict("SELECT id as item FROM rlcp WHERE id > 4600000")
    toc = time()
    logger.info(f"bornsql inference time (s): {toc-tic}")

    # Accuracy
    with engine.connect() as con:
        y_true = pd.read_sql("SELECT id, is_match FROM rlcp WHERE id > 4600000", con)
        y_true = pd.Series(y_true.is_match.values, index=y_true.id.values)
    y_true, y_pred = y_true.values, y_pred[y_true.index].values
    logger.info(f"Report bornsql: \n{classification_report(y_true=y_true, y_pred=y_pred)}")

    # Global explanation
    ex_global = bc.explain()
    ex_global = ex_global.melt(ignore_index=False).reset_index().sort_values(by='value', ascending=False)
    ex_global.columns = ['feature', 'class', 'weight']
    ex_global = ex_global.groupby('class').head(100)
    ex_global = ex_global.reset_index(drop=True).sort_values(by=['class', 'weight'], ascending=False)
    pd.DataFrame(ex_global).to_csv(os.path.join(output, "explain_global.csv"), index=False)

def scopus(engine, output):
    if not os.path.exists(output):
        os.makedirs(output)

    # Logger
    logger = logging.getLogger("scopus")
    logger.addHandler(logging.FileHandler(os.path.join(output, "scopus.log")))
    
    # Class
    sql_class = """
        SELECT id as item, asjc/100 AS class, 1.0 AS weight
        FROM publication
        """

    # Features
    sql_pubname = """
        SELECT id as item, 'pubname:'||pubname as feature, 1.0 as weight
        FROM publication
        """
    sql_pubauthor = """
        SELECT pubid as item, 'authid:'||authid as feature, 1.0 as weight
        FROM pub_author
        """
    sql_keywords = """
        SELECT pubid as item, 'keyword:'||keyword as feature, 1.0 as weight
        FROM pub_keyword 
        """
    
    # Vectorize abstract
    if engine.dialect.name == 'postgresql':
        sql_abstract = """
            SELECT id as item, 'abstract:'||lexeme as feature, array_length(positions, 1) as weight
            FROM publication, unnest(abstract)
            """
    elif engine.dialect.name == 'sqlite':
        # N_n is an internal table to select items
        sql_abstract = """
            select publication.id as item, value as feature, count(*) as weight 
            from N_n JOIN publication on N_n.item = publication.id, json_each(
                '["' || replace(publication.abstract, ' ', '","') || '"]'
            )
            group by publication.id, value
            """
    elif engine.dialect.name == 'mysql':
        # N_n is an internal table to select items
        sql_abstract = """
            select p.id as item, jt.word as feature, count(*) as weight 
            from N_n JOIN publication as p on N_n.item = p.id, json_table(
                '["' || replace(p.abstract, ' ', '","') || '"]',
                "$[*]" columns(word varchar(512) path "$")
            ) as jt
            group by p.id, jt.word
            """
    else:
        raise ValueError(f"Unsupported DBMS: {engine.dialect.name}")


    # Configurations
    configs1 = {
        'class': sql_class,    
        'features': [sql_pubname, sql_pubauthor, sql_keywords, sql_abstract]   
    }
    configs2 = {
        'class': sql_class,    
        'features': [sql_abstract]   
    }

    # Instance 1: used to partial fit
    bc1 = BornClassifierSQL(id="bc1", engine=engine, configs=configs1, type_class=Integer, type_feature=MyString)
    # Instance 2: used to fit from scratch
    bc2 = BornClassifierSQL(id="bc2", engine=engine, configs=configs1, type_class=Integer, type_feature=MyString)
    # Instance 3: used to partial fit on abstract only
    bc3 = BornClassifierSQL(id="bc3", engine=engine, configs=configs2, type_class=Integer, type_feature=MyString)
    # Instance 4: used to partial fit chronologically
    bc4 = BornClassifierSQL(id="bc4", engine=engine, configs=configs1, type_class=Integer, type_feature=MyString)

    # Timing
    rows = []
    n_items =  2359828
    example_item = "SELECT 13 as item"
    pct = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0]
    bc1.undeploy(deep=True)
    bc3.undeploy(deep=True)
    bc4.undeploy(deep=True)
    for i in range(1, len(pct)):
        logger.info(f"Start run for {pct[i]*100}% of items...")
        bc2.undeploy(deep=True)
        row = {'pct': pct[i]}

        # latency
        tic = time()
        with engine.connect() as con:
            pd.read_sql("SELECT 1", con)
        toc = time()
        row['time_latency'] = toc - tic
        
        # partial fit
        tic = time()
        bc1.partial_fit(f"""
            SELECT id as item
            FROM publication
            WHERE id % 10 = {int(pct[i-1]*10)}
            """)
        toc = time()
        row['time_fit_partial'] = toc - tic
        logger.info(f"-partial_fit: {toc - tic}")

        # partial fit abstract only
        tic = time()
        bc3.partial_fit(f"""
            SELECT id as item
            FROM publication
            WHERE id % 10 = {int(pct[i-1]*10)}
            """)
        toc = time()
        row['time_fit_partial_abstract'] = toc - tic
        logger.info(f"-partial_fit_abstract: {toc - tic}")

        # partial fit chronologic
        tic = time()
        bc4.partial_fit(f"""
            SELECT id as item
            FROM publication
            WHERE id > {int(pct[i-1]*n_items)} AND id <= {int(pct[i]*n_items)}
            """)
        toc = time()
        row['time_fit_partial_chronologic'] = toc - tic
        logger.info(f"-partial_fit_chronologic: {toc - tic}")

        # fit from scratch
        tic = time()
        bc2.fit(f"""
            SELECT id as item
            FROM publication
            WHERE id % 10 <= {int(pct[i-1]*10)}
            """)
        toc = time()
        row['time_fit'] = toc - tic
        logger.info(f"-fit: {toc - tic}")

        # number of features, classes, and items
        with engine.connect() as con:
            row['num_features'] = pd.read_sql(text(
                f"""
                SELECT COUNT(DISTINCT feature)
                FROM {bc1.db.table_corpus}            
                """), con).values[0][0]
            row['num_features_abstract'] = pd.read_sql(text(
                f"""
                SELECT COUNT(DISTINCT feature)
                FROM {bc3.db.table_corpus}            
                """), con).values[0][0]
            row['num_features_chronologic'] = pd.read_sql(text(
                f"""
                SELECT COUNT(DISTINCT feature)
                FROM {bc4.db.table_corpus}            
                """), con).values[0][0]
            row['num_classes'] = pd.read_sql(text(
                f"""
                SELECT COUNT(DISTINCT class)
                FROM {bc1.db.table_corpus}            
                """), con).values[0][0]
            row['items_new'] = pd.read_sql(text(
                f"""
                SELECT COUNT(id)
                FROM publication
                WHERE id % 10 = {int(pct[i-1]*10)}
                """), con).values[0][0]
            row['items_tot'] = pd.read_sql(text(
                f"""
                SELECT COUNT(id)
                FROM publication
                WHERE id % 10 <= {int(pct[i-1]*10)}
                """), con).values[0][0]

        # deploy time
        tic = time()
        bc2.deploy()
        toc = time()
        row['time_deploy'] = toc - tic
        logger.info(f"-deploy: {toc - tic}")

        # deploy time abstract only
        tic = time()
        bc3.deploy()
        toc = time()
        row['time_deploy_abstract'] = toc - tic
        logger.info(f"-deploy_abstract: {toc - tic}")
        bc3.undeploy(deep=False)

        # deploy time chronologic
        tic = time()
        bc4.deploy()
        toc = time()
        row['time_deploy_chronologic'] = toc - tic
        logger.info(f"-deploy_chronologic: {toc - tic}")
        bc4.undeploy(deep=False)
        
        # inference deployed
        tic = time()
        bc2.predict(example_item)
        toc = time()
        row['time_pred_deployed'] = toc - tic
        logger.info(f"-predict deployed: {toc - tic}")

        # inference undeployed
        tic = time()
        bc1.predict(example_item)
        toc = time()
        row['time_pred_undeployed'] = toc - tic
        logger.info(f"-predict undeployed: {toc - tic}")
            
        # append and log
        rows.append(row)
        logger.info(f"Row: {row}\n\n")

    # Export times to csv
    pd.DataFrame(rows).to_csv(os.path.join(output, "timing.csv"), index=False)

    # Inference deployed
    tic = time()
    bc2.predict("SELECT id as item FROM publication WHERE id <= 1000")
    toc = time()
    logger.info(f"-predict deployed (1000 items): {toc - tic}")

    # Inference undeployed
    tic = time()
    bc1.predict("SELECT id as item FROM publication WHERE id <= 1000")
    toc = time()
    logger.info(f"-predict undeployed (1000 items): {toc - tic}")
        
    # Transform
    items = bc2._transform(example_item)
    sql = f"WITH {bc2.db._sql_N_n(items)}, {bc2.db._sql_X_nj(items)} SELECT * FROM X_nj"
    with engine.connect() as con:
        transform = pd.read_sql(text(sql), con)
        pd.DataFrame(transform).to_csv(os.path.join(output, "transform.csv"), index=False)

    # Predicted class
    pred = bc2.predict(example_item).values[0]
    logger.info(f"-predicted class (example item): {pred}")

    # Actual class
    with engine.connect() as con:
        asjc = pd.read_sql(text(f"SELECT asjc FROM publication WHERE id IN ({example_item})"), con).values[0][0]
        logger.info(f"-actual class (example item): {asjc // 100}")

    # Local explanation
    ex_local = bc2.explain(example_item)
    ex_local = ex_local.melt(ignore_index=False).reset_index().sort_values(by='value', ascending=False)
    ex_local.columns = ['feature', 'class', 'weight']
    pd.DataFrame(ex_local).to_csv(os.path.join(output, "explain_local.csv"), index=False)

    # Global explanation
    ex_global = bc2.explain()
    ex_global = ex_global.melt(ignore_index=False).reset_index().sort_values(by='value', ascending=False)
    ex_global.columns = ['feature', 'class', 'weight']
    ex_global = ex_global.groupby('class').head(10)
    ex_global = ex_global.reset_index(drop=True).sort_values(by=['class', 'weight'], ascending=False)
    pd.DataFrame(ex_global).to_csv(os.path.join(output, "explain_global.csv"), index=False)

def scopus_plot_fit(output):
    data = []
    for dbms in ['PostgreSQL', 'MySQL', 'SQLite']:
        file = os.path.join('output', dbms.lower(), 'scopus', 'timing.csv')
        if os.path.exists(file):
            df = pd.read_csv(file).melt(id_vars=['pct'], value_vars=['time_fit', 'time_fit_partial'])
            df["variable"] = df["variable"].map({'time_fit': f'{dbms} - Fit', 'time_fit_partial': f'{dbms} - Partial Fit'})
            data.append(df)
    data = pd.concat(data)
    palette = {
        'PostgreSQL - Partial Fit': "#2D5986", 
        'MySQL - Partial Fit': "#862D2D", 
        'SQLite - Partial Fit': "#2D8634", 
        'PostgreSQL - Fit': "#96B9DC",
        'MySQL - Fit': "#DC9696",
        'SQLite - Fit': "#96DCB9",
    }
    fig, ax = plt.subplots(1, 1, figsize = (8,5))
    ax = sns.barplot(
        ax=ax, 
        data=data,
        x="pct", 
        y="value", 
        hue="variable", 
        palette=palette
    )
    ax.grid(axis="y", alpha=0.2)
    ax.set_ylabel('Runtime (seconds)', fontsize=15.0)
    ax.set_xlabel('Fraction of items used for training', fontsize=15.0)
    ax.tick_params(left=False) 
    ax.tick_params(bottom=False) 
    ax.get_legend().remove()
    fig.legend(fontsize=12, loc="outside upper left", ncol=3)
    sns.despine(fig=fig, ax=ax, top=True, right=True, left=True, bottom=True)
    fig.savefig(os.path.join(output, "fit.pdf"))

def scopus_plot_pred(output):
    data = []
    for dbms in ['PostgreSQL', 'MySQL', 'SQLite']:
        file = os.path.join('output', dbms.lower(), 'scopus', 'timing.csv')
        if os.path.exists(file):
            df = pd.read_csv(file).melt(id_vars=['pct'], value_vars=['time_pred_undeployed', 'time_pred_deployed'])
            df["variable"] = df["variable"].map({'time_pred_undeployed': f'{dbms} - Undeployed', 'time_pred_deployed': f'{dbms} - Deployed'})
            data.append(df)
    data = pd.concat(data)
    palette = {
        'PostgreSQL - Deployed': "#2D5986", 
        'MySQL - Deployed': "#862D2D", 
        'SQLite - Deployed': "#2D8634", 
        'PostgreSQL - Undeployed': "#96B9DC",
        'MySQL - Undeployed': "#DC9696",
        'SQLite - Undeployed': "#96DCB9",
    }
    fig, ax = plt.subplots(1, 1, figsize = (8,5))
    ax = sns.barplot(
        ax=ax,
        data=data,
        x="pct", 
        y="value", 
        hue="variable", 
        palette=palette
    )
    for bar in ax.patches[:60]:
        x = bar.get_x() + bar.get_width() / 2
        y = bar.get_height()
        ax.plot(x, y, marker='o', color=bar.get_facecolor(), markeredgecolor="black", markersize=3, zorder=5)
    ax.grid(axis="y", alpha=0.2)
    ax.set_xlabel('Fraction of items used for training', fontsize=15.0)
    ax.set_ylabel('Runtime (seconds)', fontsize=15.0)
    ax.tick_params(left=False) 
    ax.get_legend().remove()
    fig.legend(fontsize=12, loc="outside upper left", ncol=3)
    sns.despine(fig=fig, ax=ax, top=True, right=True, left=True, bottom=True)
    fig.savefig(os.path.join(output, "pred.pdf"))

def scopus_plot_deploy(output, which):
    data = []
    for dbms in ['PostgreSQL', 'MySQL', 'SQLite']:
        file = os.path.join('output', dbms.lower(), 'scopus', 'timing.csv')
        if os.path.exists(file):
            df = pd.read_csv(file)
            zero = pd.DataFrame([[0] * df.shape[1]], columns=df.columns)
            df = pd.concat([zero, df]).reset_index(drop=True)
            df['time_deploy'] = df[f'time_deploy{which}'] / df[f'time_deploy{which}'].max()
            df["variable"] = dbms
            data.append(df)
    data = pd.concat(data)
    palette = {
        'PostgreSQL': "#2D5986", 
        'MySQL': "#862D2D", 
        'SQLite': "#2D8634", 
    }
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax = sns.lineplot(
        ax=ax,
        data=data,
        x='pct',
        y='time_deploy',
        linewidth=1, 
        hue='variable',
        palette=palette,
        marker='o',
        markersize=14.0,
        markeredgewidth=2.0
    )
    ax.grid(axis="y", alpha=0.2)
    ax.set_xlabel('Fraction of items used for training', fontsize=15.0)
    ax.set_ylabel('Runtime / Max runtime', fontsize=15.0)
    ax.tick_params(left=False)
    ax.tick_params(right=False)
    ax.tick_params(bottom=False)
    ax.set_xticks(data['pct'])
    sns.despine(fig=fig, ax=ax, top=True, right=True, left=True, bottom=True)
    ax.get_legend().remove()
    fig.legend(fontsize=18, title=None, loc="outside upper center", ncol=3)
    fig.savefig(os.path.join(output, f"deploy{which}.pdf"))

def scopus_plot_features(output, which):
    data = pd.read_csv(os.path.join(output, "timing.csv"))
    zero = pd.DataFrame([[0] * data.shape[1]], columns=data.columns)
    data = pd.concat([zero, data]).reset_index(drop=True)
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax = sns.lineplot(
        ax=ax,
        x=data['pct'],
        y=data[f"num_features{which}"] // 10**3,
        linewidth=1, 
        color="#000000",
        marker='o',
        markersize=14.0,
        markeredgewidth=2.0,
        label="Number of features"
    )
    xlim = [0, 1]
    ylim = [0, data[f"num_features{which}"][1] // 10**3 * (len(data) - 1)]
    ax.plot(xlim, ylim, color='#000000', linestyle='--', linewidth=1, label="Interpolation")
    ax.grid(axis="y", alpha=0.2)
    ax.set_xlabel('Fraction of items used for training', fontsize=15.0)
    ax.set_ylabel('N. features (thousands)', fontsize=15.0)
    ax.set_xticks(data['pct'])
    ax.tick_params(left=False)
    ax.tick_params(right=False)
    ax.tick_params(bottom=False)
    sns.despine(fig=fig, ax=ax, top=True, right=True, left=True, bottom=True)
    ax.get_legend().remove()
    fig.legend(fontsize=18, title=None, loc="outside upper center", ncol=3)
    fig.savefig(os.path.join(output, f"features{which}.pdf"))

def scopus_plot_deploy_vs_features(output, which):
    data = []
    for dbms in ['PostgreSQL', 'MySQL', 'SQLite']:
        file = os.path.join('output', dbms.lower(), 'scopus', 'timing.csv')
        if os.path.exists(file):
            df = pd.read_csv(file)
            zero = pd.DataFrame([[0] * df.shape[1]], columns=df.columns)
            df = pd.concat([zero, df]).reset_index(drop=True)
            df["num_features"] = df[f'num_features{which}'] // 10**3
            df['time_deploy'] = df[f'time_deploy{which}'] / df[f'time_deploy{which}'].max()
            df["variable"] = dbms
            data.append(df)
    data = pd.concat(data)
    palette = {
        'PostgreSQL': "#2D5986", 
        'MySQL': "#862D2D", 
        'SQLite': "#2D8634", 
    }
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax = sns.lineplot(
        ax=ax,
        data=data,
        x='num_features',
        y='time_deploy',
        linewidth=0.3, 
        hue='variable',
        palette=palette,
        marker='o',
        markersize=10.0,
        markeredgewidth=2.0
    )
    ax.grid(axis="y", alpha=0.2)
    ax.set_xlabel('N. features (thousands)', fontsize=15.0)
    ax.set_ylabel('Runtime / Max runtime', fontsize=15.0)
    ax.tick_params(left=False)
    ax.tick_params(right=False)
    ax.tick_params(bottom=False)
    sns.despine(fig=fig, ax=ax, top=True, right=True, left=True, bottom=True)
    ax.get_legend().remove()
    fig.legend(fontsize=12, title=None, loc="outside upper center", ncol=3)
    fig.savefig(os.path.join(output, f"deploy_vs_features{which}.pdf"))

def neurips(engine, output, dataset):
    if not os.path.exists(output):
        os.makedirs(output)
    
    data = Dataset(dataset=dataset, output_dir="data")

    bc = BornClassifierSQL(engine=engine, id=f"neurips_{dataset}", type_class=MyString, type_feature=MyString)
    bc.undeploy(deep=True)
    bc.fit(data.X_train, data.y_train)
    y_pred = bc.predict(data.X_test)

    logger = logging.getLogger(dataset)
    logger.addHandler(logging.FileHandler(os.path.join(output, f"{dataset}.log")))
    logger.info(f"Accuracy = {(data.y_test == y_pred).mean()}")
    

if __name__ == "__main__":
    args = parse_args()

    # Create an engine to the database
    if args.dbms == 'postgresql':
        MyString = String(512)
        engine = create_engine(
            url=f"postgresql+psycopg2://{args.user}:{args.password}@{args.host}:{args.port}/{args.database}",
            executemany_mode='values_plus_batch',
            insertmanyvalues_page_size=1000,
            executemany_batch_page_size=1000
        )
    elif args.dbms == 'mysql':
        MyString = String(512, collation='utf8mb4_bin')
        engine = create_engine(
            url=f"mysql+pymysql://{args.user}:{args.password}@{args.host}:{args.port}/{args.database}?sql_mode=PIPES_AS_CONCAT",
            insertmanyvalues_page_size=1000
        )
    elif args.dbms == 'sqlite':
        MyString = String(512)
        engine = create_engine(
            url=f"sqlite:///{args.database}.db"
        )
    else:
        raise ValueError(f"Unsupported DBMS: {args.dbms}")

    # Run scopus on all DBMS
    print("Running scopus...")
    output = f'output/{args.dbms}/scopus'
    scopus(engine, output)
    scopus_plot_fit(output)
    scopus_plot_pred(output)
    scopus_plot_deploy(output, which='')
    scopus_plot_deploy(output, which='_abstract')
    scopus_plot_deploy(output, which='_chronologic')
    scopus_plot_features(output, which='')
    scopus_plot_features(output, which='_abstract')
    scopus_plot_features(output, which='_chronologic')
    scopus_plot_deploy_vs_features(output, which='')

    # Run all datasets only on PostgreSQL
    if args.dbms == 'postgresql':
        # NG dataset
        print("Running 20 newsgroups...")
        output = f'output/{args.dbms}/20ng'
        neurips(engine, output, dataset='20ng')

        # R52 dataset
        print("Running R52...")
        output = f'output/{args.dbms}/r52'
        neurips(engine, output, dataset='r52')

        # R8 dataset
        print("Running R8...")
        output = f'output/{args.dbms}/r8'
        neurips(engine, output, dataset='r8')

        # Adult dataset
        print("Running adult...")
        output = f'output/{args.dbms}/adult'
        adult(engine, output)

        # RLCP dataset
        print("Running rlcp...")
        output = f'output/{args.dbms}/rlcp'
        rlcp(engine, output)
