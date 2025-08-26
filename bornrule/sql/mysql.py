from .database import Database
from sqlalchemy import text


class MySQL(Database):

    POW = 'POWER'

    CONCAT_FUN = 'CONCAT'
    CONCAT_SEP = ''

    @staticmethod
    def insert_or_ignore(con, table, values, columns=None):
        from_select = isinstance(values, str)
        keys = columns if from_select else values[0].keys()
        records = values if from_select else f"VALUES(:{', :'.join(keys)})"
        params = None if from_select else values
        sql = f"""
            INSERT IGNORE INTO {table} ({','.join(keys)})
            {records}
            """

        return con.execute(text(sql), params)

    @staticmethod
    def insert_or_replace(con, table, values, conflict, replace, columns=None):
        from_select = isinstance(values, str)
        keys = columns if from_select else values[0].keys()
        records = values if from_select else f"VALUES(:{', :'.join(keys)})"
        params = None if from_select else values
        sql = f"""
            INSERT INTO {table} ({','.join(keys)})
            {records}
            ON DUPLICATE KEY 
            UPDATE {','.join([f'{r} = :{r}' for r in replace])}
            """

        return con.execute(text(sql), params)

    @staticmethod
    def insert_or_sum(con, table, values, conflict, sum, columns=None):
        from_select = isinstance(values, str)
        keys = columns if from_select else values[0].keys()
        records = values if from_select else f"VALUES(:{', :'.join(keys)})"
        params = None if from_select else values
        sql = f"""
            INSERT INTO {table} ({','.join(keys)})
            {records}
            ON DUPLICATE KEY 
            UPDATE {", ".join([f"{s} = {table}.{s} + VALUES({s})" for s in sum])}
            """

        return con.execute(text(sql), params)
