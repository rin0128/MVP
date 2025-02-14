from langchain_neo4j import Neo4jGraph
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

# Neo4jに接続
graph = Neo4jGraph(
    url=NEO4J_URI,
    username=NEO4J_USER,
    password=NEO4J_PASSWORD
)

# すでにデータがある場合は一旦削除（オプション）
#graph.query("MATCH (n) DETACH DELETE n")

# クエリを実行
query = """
CREATE (nakao:Person {
    name: '中尾',
    age: 21,
    university_year: 3,
    mbti: 'ENTP',
    gender: '男',
    field: '文系',
    major: '経営学部',
    university: '京都産業大学'
})

CREATE (freedom:Value {name: '自由'})

CREATE (philosophy:Interest {name: '哲学'})
CREATE (soccer:Interest {name: 'サッカー'})
CREATE (exercise:Interest {name: '運動'})

CREATE (mbti:PersonalityType {name: 'ENTP'})

CREATE (entrepreneur:Occupation {name: '起業家'})
CREATE (ceo:Occupation {name: '経営者'})

CREATE (dream:Goal {name: '世界を変えること'})

CREATE (ai_sns:Field {name: 'パーソナルAI × SNS'})

CREATE (affiliation:Experience {name: 'アフェリエイト', started: '高校'})
CREATE (sns_management:Experience {name: 'SNS運用代行', started: '大学'})
CREATE (event_management:Experience {name: 'イベント運営', started: '大学'})
CREATE (soccer_past:Experience {name: '小中高サッカー', started: '小学校', ended: '高校'})

CREATE (nakao)-[:LIKES]->(philosophy)
CREATE (nakao)-[:LIKES]->(soccer)
CREATE (nakao)-[:LIKES]->(exercise)

CREATE (nakao)-[:HAS_PERSONALITY]->(mbti)

CREATE (nakao)-[:PURSUING]->(dream)
CREATE (dream)-[:RELATED_TO]->(ai_sns)

CREATE (nakao)-[:HAS_OCCUPATION]->(entrepreneur)
CREATE (nakao)-[:HAS_OCCUPATION]->(ceo)

CREATE (nakao)-[:HAS_EXPERIENCE]->(affiliation)
CREATE (nakao)-[:HAS_EXPERIENCE]->(sns_management)
CREATE (nakao)-[:HAS_EXPERIENCE]->(event_management)
CREATE (nakao)-[:HAS_EXPERIENCE]->(soccer_past)

CREATE (nakao)-[:VALUES]->(freedom)

"""

graph.query(query)

print("✅ データの挿入が完了しました！")
