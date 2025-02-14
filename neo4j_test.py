from langchain_neo4j import Neo4jGraph
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

graph = Neo4jGraph(
    url=NEO4J_URI,
    username=NEO4J_USER,
    password=NEO4J_PASSWORD
)

try:
    result = graph.query("RETURN 'Connection Successful' AS message")
    print("✅ Neo4j Connection Successful:", result)
except Exception as e:
    print("❌ Connection Failed:", e)


