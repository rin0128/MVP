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

# ======================
# 1. Cypherクエリ生成プロンプト
# ======================
cypher_template = """Neo4jの以下のグラフスキーマに基づいて、ユーザーの質問に答えるCypherクエリを書いてください。
スキーマ: {schema}
質問: {question}

以下に注意してください:
- クエリ以外の文章は出力しないでください。
- 質問がグラフと無関係、もしくはクエリ生成が不可能な場合は「NO_QUERY」だけを返してください。

Cypherクエリ:"""

cypher_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "入力された質問をCypherクエリに変換してください。クエリ以外は生成しないでください。"),
        ("human", cypher_template),
    ]
)

# ======================
# 2. OpenAI LLM のセットアップ
# ======================
# 例として gpt-4 を使用。model 引数や温度を調整してください。
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=1,
    openai_api_key=OPENAI_API_KEY
)

# ======================
# 3. クエリ生成チェイン
# ======================
queryGenChain = (
    RunnablePassthrough.assign(schema=lambda _: graph.get_schema)  # graph.get_schema を schema にセット
    | cypher_prompt
    | llm
    | StrOutputParser()
)

# ======================
# 4. クエリ実行 & 自然言語変換のチェイン
# ======================
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

# ======================
# 5. 全体チェイン
# ======================
chain = (
    RunnablePassthrough.assign(query=queryGenChain)  
    | RunnablePassthrough.assign(
        response=lambda x: execute_cypher_or_none(
            query=x["query"],
            graph=graph
        )
    )
    | response_prompt
    | llm
    | StrOutputParser()
)

def execute_cypher_or_none(query: str, graph: Neo4jGraph):
    """
    - 受け取ったクエリが 'NO_QUERY' なら None を返す
    - 正規表現で ```cypher などを除去して実行
    - 実行結果が空なら空のリスト/Noneを返す
    """
    # "NO_QUERY" が返ってきた場合 → グラフと無関係な質問
    if query.strip().upper() == "NO_QUERY":
        return None
    
    # 不要なマークダウン表記を除去
    cleaned_query = re.sub(r"```(cypher)?|```", "", query).strip()
    
    if not cleaned_query:
        # 生成されたクエリが空文字列の場合
        return None
    
    try:
        results = graph.query(cleaned_query)
        # クエリ実行結果が空のリスト/辞書の場合は None を返す
        if not results:
            return None
        return results
    except Exception as e:
        # Neo4jクエリ実行時のエラーをキャッチしてログを残し、None を返す
        print(f"Cypher実行エラー: {e}")
        return None

# ======================
# 6. メイン処理
# ======================
if __name__ == "__main__":
    # 例: グラフと無関係な質問
    question = "私は中尾です。私の魅力について初対面の人にわかるように表現してください？"
    # question = "田中さんが所属している部署を教えてください。(グラフにある情報)"
    
    result = chain.invoke({"question": question})
    print("🟢 AI の回答:", result)
