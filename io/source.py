import pymysql

pymysql.install_as_MySQLdb()


class SQLSource:
    DBAPI_DICT = {
        'mysql': 'mysqldb',
        'mssql': 'pymssql'
    }

    def __init__(self, params):
        db_type = params['db_type']
        if db_type not in SQLSource.DBAPI_DICT:
            raise ValueError('database type {} is not provided\nUse one of the following types: {}'.format(
                db_type, str(list(SQLSource.DBAPI_DICT.keys()))))
        dbapi = SQLSource.DBAPI_DICT[db_type]
        prefix = '+'.join([db_type, dbapi]) if len(dbapi) > 0 else db_type
        self._connection_config = '{}://{}:{}@{}/{}'.format(prefix, params['username'], params['password'],
                                                            params['hostname'], params['database'])

    def get_data_since(self, table, date=None):
        return

    def find_last_date(self, table):
        return

    def write_new_data(self, table, data):
        return

    def clean_old_data(self, table, keep_period):
        return
