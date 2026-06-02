from flask import Flask, request
import pandas as pd
import ollama
import chromadb
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

df = pd.read_csv("superstore.csv")

total_sales = df["Sales"].sum()
total_orders = df["Order ID"].nunique()

top_region = df.groupby("Region")["Sales"].sum().sort_values(ascending=False).index[0]
top_category = df.groupby("Category")["Sales"].sum().sort_values(ascending=False).index[0]

documents = [
    f"Total sales in the retail dataset are {total_sales:.2f}.",
    f"The dataset contains {total_orders} unique orders.",
    f"The top region by sales is {top_region}.",
    f"The top category by sales is {top_category}."
]

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedding_model.encode(documents).tolist()

client = chromadb.Client()
collection = client.get_or_create_collection(name="retail_rag_demo")

try:
    collection.add(
        documents=documents,
        embeddings=embeddings,
        ids=[f"doc_{i}" for i in range(len(documents))]
    )
except:
    pass


def ask_rag(question):

    question_embedding = embedding_model.encode([question]).tolist()

    results = collection.query(
        query_embeddings=question_embedding,
        n_results=2
    )

    retrieved_docs = results["documents"][0]

    context = "\n".join(retrieved_docs)

    prompt = f"""
You are a simple retail analytics assistant.

Use only the context below.
If the answer is not available in the context,
say "I do not have enough information."

Context:
{context}

Question:
{question}

Answer in simple business language.
"""

    response = ollama.chat(
        model="llama3.2",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response["message"]["content"]


@app.route("/", methods=["GET", "POST"])
def home():
    answer = ""

    if request.method == "POST":
        question = request.form["question"]
        answer = ask_rag(question)

    return f"""
    <html>
    <body style="font-family: Arial; margin: 40px;">
        <h1>Super Store Assistant</h1>

        <form method="POST">
            <input type="text" name="question" 
                   style="width: 500px; padding: 8px;"
                   placeholder="Ask a retail analytics question">
            <button type="submit" style="padding: 8px;">Ask</button>
        </form>

        <h3>Answer:</h3>
        <p>{answer}</p>
    </body>
    </html>
    """


if __name__ == "__main__":
    app.run(debug=True)