import cx_Oracle
import db_config
import os
# Single Connetion to DB
# connection = None
# try:
#     connection = cx_Oracle.connect(
#         config.username,
#         config.password,
#         config.dsn,
#         encoding=config.encoding)

#     # show the version of the Oracle Database
#     print("Connected to Database.")
#     cursor = connection.cursor()
#     print("Query Starts.")
#     cursor.execute("""
#         SELECT COUNT(*) FROM WIND.CHINAOPTIONEODPRICES
#     """)
#     for row in cursor:
#         print(row[0])
#     print("Query Ends.")

# except cx_Oracle.Error as error:
#     print(error)
# finally:
#     # release the connection
#     if connection:
#         connection.close()

# Pooled Connection to DB

# Create the session pool


# # Acquire a connection from the pool
# connection = pool.acquire()

# # Use the pooled connection
# print('Using the connection')
# print('Query Starats')
# cursor = connection.cursor()
# cursor.execute("""
#     SELECT COUNT(*) FROM WIND.CHINAOPTIONEODPRICES
# """)
# for row in cursor:
#     print(row[0])
# print('Query Ends')

# # Release the connection to the pool
# pool.release(connection)

# # Close the pool
# pool.close()


#
# class DB_connect():

#     connections = [] 
#     connection_num = -1 
#     dbPool = None


#     def __init__(self):
#         self.poolDB()
    
#     def __del__(self):
#         self.releaseAllConnection()
#         self.closeDB()

        
#     def poolDB(self):
#         cx_Oracle.init_oracle_client(lib_dir=os.environ.get("HOME")+"/Downloads/instantclient_19_8")
#         pool = cx_Oracle.SessionPool(
#             config.username,
#             config.password,
#             config.dsn,
#             min=2,
#             max=5,
#             increment=1,
#             encoding=config.encoding
#         )
#         self.dbPool = pool
    

#     def acquireDBCursor(self):
#         self.connections.append(self.dbPool.acquire())
#         self.connection_num += 1
#         print("Acquiring a new DB connection...")
#         return self.connections[self.connection_num]

#     def executeSQL(self, sql):
#         c = self.acquireDBCursor().cursor()
#         c.execute(sql)
#         return c 

#     def releaseAllConnection(self):
#         for i in range(self.connection_num + 1):
#             print(f"Releasing DB connection {i}...")
#             self.dbPool.release(self.connections[i])

#     def closeDB(self):
#         print("Closing DB Pool...")
#         self.dbPool.close()

class DB_connect():
    
    connection = None

    def __init__(self):
        self.initOracleClient()
        self.connectDB()
    
    def __del__(self):
        self.closeConnection()
    
    def initOracleClient(self):
        try:
            cx_Oracle.init_oracle_client(lib_dir=os.environ.get("HOME")+"/Downloads/instantclient_19_8") 
        except:
            print("Oracle Client has started.")

    def connectDB(self):
        try:
            connection = cx_Oracle.connect(
                db_config.username,
                db_config.password,
                db_config.dsn,
                encoding=db_config.encoding)
            print("Connected to Database.")
            self.connection = connection
        except cx_Oracle.Error as error:
            print(error)
    
    def closeConnection(self):
        if self.connection:
            self.connection.close()
            print("Connection to Database closed.")

    def executeSQL(self, sql):
        cursor = self.connection.cursor()
        print("Query starts >>>")
        cursor.execute(sql)
        res = []
        for row in cursor:
            res.append(row)
        print("Query ends <<<")
        return res

    # try:
#     connection = cx_Oracle.connect(
#         config.username,
#         config.password,
#         config.dsn,
#         encoding=config.encoding)

#     # show the version of the Oracle Database
#     print("Connected to Database.")
#     cursor = connection.cursor()
#     print("Query Starts.")
#     cursor.execute("""
#         SELECT COUNT(*) FROM WIND.CHINAOPTIONEODPRICES
#     """)
#     for row in cursor:
#         print(row[0])
#     print("Query Ends.")

# except cx_Oracle.Error as error:
#     print(error)
# finally:
#     # release the connection
#     if connection:
#         connection.close()
 
    


        