from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.graphs import Neo4jGraph
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, OPENAI_API_KEY
import re


# Neo4j に接続
graph = Neo4jGraph(
    url=NEO4J_URI,
    username=NEO4J_USER,
    password=NEO4J_PASSWORD
)



# クエリ生成のプロンプト
cypher_template = """Neo4jの以下のグラフスキーマに基づいて、ユーザーの質問に答えるCypherクエリを書いてください。
スキーマ: {schema}
質問: {question}
Cypherクエリ:"""

cypher_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "入力された質問をCypherクエリに変換してください。クエリ以外は生成しないでください。"),
        ("human", cypher_template),
    ]
)

# OpenAI LLM のセットアップ
llm = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=OPENAI_API_KEY)

# クエリ生成チェイン
queryGenChain = (
    RunnablePassthrough.assign(schema=lambda _: graph.get_schema)  
    | cypher_prompt  
    | llm  
    | StrOutputParser()  
)

# クエリ実行 & 自然言語変換のチェイン
response_template = """質問、Cypherクエリ、およびクエリ実行結果に基づいて、自然言語で回答を書いてください:
質問: {question}
Cypherクエリ: {query}
クエリ実行結果: {response}"""

response_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "入力された質問、クエリ、クエリ実行結果をもとに、自然言語の答えに変換してください。"),
        ("human", response_template),
    ]
)

# ✅ クエリ実行 & 自然言語変換のチェイン
chain = (
    RunnablePassthrough.assign(query=queryGenChain)  # クエリを生成
    | RunnablePassthrough.assign(
        response=lambda x: graph.query(re.sub(r"```cypher|```", "", x["query"]).strip())  # クエリを実行
    )
    | response_prompt  # LLM に自然言語で回答させる
    | llm
    | StrOutputParser()  # 最終的な回答を文字列にする
)

# ✅ 質問を入力
if __name__ == "__main__":
    result = chain.invoke({"question": "中尾の好きなものは？"})
    print("🟢 AI の回答:", result)
