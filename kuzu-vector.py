# pip install sentence-transformers
import kuzu
import os

db = kuzu.Database("example-2.kuzu")
conn = kuzu.Connection(db)

from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")

conn.execute("INSTALL vector; LOAD vector;")

conn.execute("CREATE NODE TABLE Book(id SERIAL PRIMARY KEY, title STRING, title_embedding FLOAT[384], published_year INT64);")
conn.execute("CREATE NODE TABLE Publisher(name STRING PRIMARY KEY);")
conn.execute("CREATE REL TABLE PublishedBy(FROM Book TO Publisher);")

titles = [
    "The Quantum World",
    "Chronicles of the Universe",
    "Learning Machines",
    "Echoes of the Past",
    "The Dragon's Call"
]
publishers = ["Harvard University Press", "Independent Publisher", "Pearson", "McGraw-Hill Ryerson", "O'Reilly"]
published_years = [2004, 2022, 2019, 2010, 2015]

for title, published_year in zip(titles, published_years):
    embeddings = model.encode(title).tolist()
    conn.execute(
        """
        CREATE (b:Book {
            title: $title,
            title_embedding: $embeddings,
            published_year: $year
        });""",
        {"title": title, "year": published_year, "embeddings": embeddings}
    )

    print(f"Inserted book: {title}")

for publisher in publishers:
    conn.execute(
        """CREATE (p:Publisher {name: $publisher});""",
        {"publisher": publisher}
    )
    print(f"Inserted publisher: {publisher}")

for title, publisher in zip(titles, publishers):
    conn.execute("""
        MATCH (b:Book {title: $title})
        MATCH (p:Publisher {name: $publisher})
        CREATE (b)-[:PublishedBy]->(p);
        """,
        {"title": title, "publisher": publisher}
    )
    print(f"Created relationship between {title} and {publisher}")