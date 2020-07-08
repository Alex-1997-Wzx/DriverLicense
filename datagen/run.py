''' 将图片存放到mymodel.h5文件中，隐藏图片...'''
import sqlite3

root_dir = './gen/original'

def convertToBinaryData(filename):
    with open(filename, 'rb') as file:
        blobData = file.read()
    return blobData


def create_table():
    sqlstr = '''CREATE TABLE IF NOT EXISTS `newtable` (
                    `name` TEXT DEFAULT NULL,
                    `photo` BLOB DEFAULT NULL
                    )'''
    sqliteConnection = sqlite3.connect('mymodel')
    cursor = sqliteConnection.cursor()
    cursor.execute(sqlstr)
    sqliteConnection.commit()
    cursor.close()
    sqliteConnection.close()

def insertBLOB(name, photo):
    sqliteConnection = sqlite3.connect('mymodel')
    cursor = sqliteConnection.cursor()
    try:
        sqlite_insert_blob_query = """ INSERT INTO 'newtable'
                                  ('name', 'photo') VALUES (?, ?)"""

        empPhoto = convertToBinaryData(photo)
        data_tuple = (name, empPhoto)
        cursor.execute(sqlite_insert_blob_query, data_tuple)
        sqliteConnection.commit()
        print("inserted successfully as a BLOB into a table")
        cursor.close()

    except sqlite3.Error as error:
        print("Failed to insert blob data into sqlite table", error)
    finally:
        if (sqliteConnection):
            sqliteConnection.close()
            print("the sqlite connection is closed")


def writeTofile(data, filename):
    with open(filename, 'wb') as file:
        file.write(data)
    print("Stored blob data into: ", filename, "\n")


def readBlobData():
    try:
        sqliteConnection = sqlite3.connect('mymodel.h5')
        cursor = sqliteConnection.cursor()
        print("Connected to SQLite")

        sql_fetch_blob_query = "SELECT * FROM newtable;"
        # sql_fetch_blob_query = "SELECT * FROM newtable LIMIT 100;"
        cursor.execute(sql_fetch_blob_query)
        record = cursor.fetchall()
        for row in record:
            name  = row[0]
            photo = row[1]
            photoPath = os.path.join('kaga', str(name))
            writeTofile(photo, photoPath)

        cursor.close()

    except sqlite3.Error as error:
        print("Failed to read blob data from sqlite table", error)
    finally:
        if (sqliteConnection):
            sqliteConnection.close()
            print("sqlite connection is closed")


if __name__ == '__main__':
    import os
    # create_table()
    # for f in os.listdir(root_dir):
    #     if len(f) < 4:
    #         continue
    #     if f[-4:] not in ['.jpg', '.png', '.JPG', '.PNG', 'jpeg', 'JPEG']:
    #         continue
    #     fname = os.path.join(root_dir, f)
    #     insertBLOB(f, fname)
    readBlobData()