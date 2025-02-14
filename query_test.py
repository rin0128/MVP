from langchain_neo4j import Neo4jGraph
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

# Neo4jに接続
graph = Neo4jGraph(
    url=NEO4J_URI,
    username=NEO4J_USER,
    password=NEO4J_PASSWORD
)

# クエリテスト1: 中尾が好きなものを取得
query1 = """
MATCH (nakao:Person {name: '中尾'})-[:LIKES]->(thing)
RETURN thing.name AS FavoriteThings
"""
result1 = graph.query(query1)
print("✅ 中尾の好きなもの:", result1)

# クエリテスト2: 中尾の職業を取得
query2 = """
MATCH (nakao:Person {name: '中尾'})-[:HAS_OCCUPATION]->(job)
RETURN job.name AS Occupation
"""
result2 = graph.query(query2)
print("✅ 中尾の職業:", result2)

# クエリテスト3: 中尾の経験を取得
query3 = """
MATCH (nakao:Person {name: '中尾'})-[:HAS_EXPERIENCE]->(exp)
RETURN exp.name AS Experience, exp.started AS Started
"""
result3 = graph.query(query3)
print("✅ 中尾の経験:", result3)

# クエリテスト4: 中尾の夢と関連する分野を取得
query4 = """
MATCH (nakao:Person {name: '中尾'})-[:PURSUING]->(dream)-[:RELATED_TO]->(field)
RETURN dream.name AS Dream, field.name AS RelatedField
"""
result4 = graph.query(query4)
print("✅ 中尾の夢:", result4)
