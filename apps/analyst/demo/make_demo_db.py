"""Synthetic demo data for Data Desk (a small bookshop), so the app can be
tried without private data.

    python -m apps.analyst.demo.make_demo_db ~/edgeinfer-data/demo/shop.db
    python -m apps.analyst.build_index apps/analyst/demo/demo.json
"""
import random
import sqlite3
import sys
from pathlib import Path

GENRES = ["fantasy", "history", "science", "cooking", "travel", "poetry", "business", "children"]
CITIES = ["Delhi", "Mumbai", "Pune", "Jaipur", "Faridabad", "Bengaluru", "Kolkata", "Chennai"]
ADJ = ["quiet", "bright", "lost", "hidden", "golden", "northern", "last", "small", "wild", "patient"]
NOUN = ["river", "garden", "kitchen", "empire", "harbour", "lantern", "map", "orchard", "theory", "market"]
PRAISE = ["Loved the pacing", "Beautifully written", "Great gift", "Clear explanations", "Recipes actually work",
          "My kids ask for it every night", "Changed how I think about money", "Perfect for a long train ride"]
GRIPE = ["arrived with a torn cover", "too long in the middle", "the print is small", "ending felt rushed",
         "expected more photos", "translation is clumsy"]


def main(out: str) -> None:
    rnd = random.Random(7)
    p = Path(out).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.unlink(missing_ok=True)
    c = sqlite3.connect(p)
    c.executescript("""
        CREATE TABLE books(id INTEGER PRIMARY KEY, title TEXT, genre TEXT, price_inr INTEGER, pages INTEGER, year INTEGER);
        CREATE TABLE customers(id INTEGER PRIMARY KEY, name TEXT, city TEXT, joined TEXT);
        CREATE TABLE orders(id INTEGER PRIMARY KEY, customer_id INTEGER, book_id INTEGER, qty INTEGER, day TEXT);
        CREATE TABLE reviews(id INTEGER PRIMARY KEY, book_id INTEGER, customer_id INTEGER, stars INTEGER, text TEXT);
    """)
    for i in range(1, 121):
        g = rnd.choice(GENRES)
        c.execute("INSERT INTO books VALUES (?,?,?,?,?,?)", (i, f"The {rnd.choice(ADJ).title()} {rnd.choice(NOUN).title()}",
                  g, rnd.randrange(199, 1499), rnd.randrange(80, 700), rnd.randrange(1990, 2026)))
    for i in range(1, 401):
        c.execute("INSERT INTO customers VALUES (?,?,?,?)", (i, f"customer{i:03d}", rnd.choice(CITIES),
                  f"202{rnd.randrange(0, 6)}-{rnd.randrange(1, 13):02d}-01"))
    for i in range(1, 3001):
        c.execute("INSERT INTO orders VALUES (?,?,?,?,?)", (i, rnd.randrange(1, 401), rnd.randrange(1, 121),
                  rnd.choice([1, 1, 1, 2, 3]), f"2026-{rnd.randrange(1, 11):02d}-{rnd.randrange(1, 29):02d}"))
    for i in range(1, 801):
        stars = rnd.choice([5, 5, 4, 4, 4, 3, 2, 1])
        text = rnd.choice(PRAISE) + "." if stars >= 4 else f"{rnd.choice(PRAISE)}, but it {rnd.choice(GRIPE)}."
        c.execute("INSERT INTO reviews VALUES (?,?,?,?,?)", (i, rnd.randrange(1, 121), rnd.randrange(1, 401), stars, text))
    c.commit()
    print(f"wrote {p}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "~/edgeinfer-data/demo/shop.db")
