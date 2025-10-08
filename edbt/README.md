# Paper replication

Install all dependencies

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

All results are stored into an `output/` folder that is generated in the working directory.


## Cite as

>Emanuele Guidotti, Darya Shlyk, Stefano Montanelli, and Alfio Ferrara. In-Database Text Classification with BornSQL. *Proceedings of the 29th international conference on extending database technology*, forthcoming, 2026.

A BibTeX entry for LaTeX users is:

```bibtex
@inproceedings{guidotti2026bornsql,
  author    = {Emanuele Guidotti and Darya Shlyk and Stefano Montanelli and Alfio Ferrara},
  title     = {In-Database Text Classification with BornSQL},
  booktitle = {Proceedings of the 29th International Conference on Extending Database Technology (EDBT)},
  year      = {2026},
  note      = {Forthcoming}
}
```