import sqlite3

conn = sqlite3.connect("visitors.db")
cursor = conn.cursor()

cursor.execute("SELECT * FROM visitors")

rows = cursor.fetchall()

for row in rows:
    print(row)

conn.close()