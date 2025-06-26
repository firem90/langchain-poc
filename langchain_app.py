from flask import Flask, request, render_template, jsonify
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

app = Flask(__name__)

# LangChain with local Ollama model
llm = Ollama(model="mistral")

# Prompt for chatbot behavior
prompt = PromptTemplate.from_template("""
You are ManishBot, a secure, business-savvy assistant. Stay formal and avoid jokes.

User: {question}
Assistant:
""")

chain = LLMChain(llm=llm, prompt=prompt)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("message", "").strip()

    if not question:
        return jsonify({"error": "Empty question"}), 400

    try:
        response = chain.run(question=question)
        return jsonify({"response": response.strip()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=8000)
