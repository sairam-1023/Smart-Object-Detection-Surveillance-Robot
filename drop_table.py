import sqlite3

conn = sqlite3.connect("visitors.db")
cursor = conn.cursor()

cursor.execute("DELETE FROM visitors")

conn.commit()
conn.close()

print("✅ All records deleted from visitors table")