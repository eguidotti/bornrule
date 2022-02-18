from .database import Database
import numpy as np
import sqlite3


class SQLite(Database):

    LOG = 'LOG'
    SUM = 'SUM'
    POW = 'POW'

    def __init__(self, **kwargs):

        # fix "Integers becomes blob" on insert. See https://sqlite.org/forum/info/fc008fb3c5a0ee97
        sqlite3.register_adapter(np.int32, int)
        sqlite3.register_adapter(np.int64, int)

        # create functions
        conn = kwargs.get('conn')
        conn.connection.create_function("POW", 2, pow)
        conn.connection.create_function("LOG", 1, np.log)

        # init parent
        super().__init__(**kwargs)
