# Paper replication

All results in the paper are obtained using Python 3.11.5 and PostgreSQL 12 with MADlib 2.1.0, MySQL 9.4.0, and SQLite 3.50.2 on a Ubuntu 20.04.6 LTS VM server with 32GB RAM and 20 assigned CPU cores. Before proceeding, download the [data](https://doi.org/10.7910/DVN/97LXCA) and unzip them in a folder named `data` in the this directory. 

Install all dependencies with [poetry](https://python-poetry.org)

```
poetry install
```

Restore the PostgreSQL database

```
poetry run python restore.py --user postgres --password <pwd> --host <host> --database <dbname>
```

Reproduce the results using BornSQL on PostgreSQL

```
poetry run python bornsql.py --user postgres --password <pwd> --host <host> --database <dbname>
```

Reproduce the results using MADlib on PostgreSQL

```
poetry run python madlib.py --user postgres --password <pwd> --host <host> --database <dbname>
```

Restore the MySQL database

```
poetry run python restore.py --user <user> --password <pwd> --host <host> --database <dbname> --dbms mysql
```

Reproduce the results using BornSQL on MySQL

```
poetry run python bornsql.py --user <user> --password <pwd> --host <host> --database <dbname> --dbms mysql
```

Restore the SQLite database

```
poetry run python restore.py --database <dbname> --dbms sqlite
```

Reproduce the results using BornSQL on SQLite

```
poetry run python bornsql.py --database <dbname> --dbms sqlite
```

All results are stored into an `output` folder that is generated in the working directory.


## Cite as

>Emanuele Guidotti, Darya Shlyk, Stefano Montanelli, and Alfio Ferrara. In-Database Text Classification with BornSQL. *Proceedings of the 29th international conference on extending database technology*, forthcoming, 2026.

A BibTeX entry for LaTeX users is:

```bibtex
@inproceedings{guidotti2026bornsql,
  author    = {Guidotti, Emanuele and Shlyk, Darya and Montanelli, Stefano and Ferrara, Alfio},
  title     = {In-Database Text Classification with BornSQL},
  booktitle = {Proceedings of the 29th International Conference on Extending Database Technology (EDBT)},
  year      = {2026},
  note      = {Forthcoming}
}
```
