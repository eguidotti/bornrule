# README

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
