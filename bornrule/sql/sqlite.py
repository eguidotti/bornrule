import sqlite3
import numpy as np
from .database import Database


class SQLite(Database):

    # Fix "Integers becomes blob" on insert
    # See https://sqlite.org/forum/info/fc008fb3c5a0ee97
    sqlite3.register_adapter(np.int32, int)
    sqlite3.register_adapter(np.int64, int)

    def connect(self):
        con = self.engine.connect()
        con.connection.create_function(self.POW, 2, pow)
        con.connection.create_function(self.LOG, 1, np.log)
        return con
