import sqlite3

conn = sqlite3.connect("visitors.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS visitors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    purpose TEXT,
    person TEXT,
    decision TEXT,
    image_path TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
""")

conn.commit()
conn.close()

print("Database created successfully")