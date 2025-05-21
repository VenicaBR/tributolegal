from flask import Flask, request, jsonify, render_template, session
from flask_cors import CORS
from openai import OpenAI
from dotenv import load_dotenv
from uuid import uuid4
import os
from pathlib import Path
import faiss
import pickle
import tiktoken
from PyPDF2 import PdfReader
import numpy as np

# ==== Configuração ====
load_dotenv()
dotenv_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path)

print("OPENAI_API_KEY lido:", os.getenv("OPENAI_API_KEY"))
app = Flask(__name__, static_folder='static', template_folder='templates')
app.secret_key = os.getenv("SECRET_KEY", "chave_secreta_padrao")
CORS(app)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

conversas = {}

# ==== Funções utilitárias ====

def chunks(lst, n):
    """Divide a lista em pedaços de tamanho n."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def carregar_texto_dos_pdfs(pasta):
    textos = []
    for nome_arquivo in os.listdir(pasta):
        if nome_arquivo.endswith(".pdf"):
            caminho = os.path.join(pasta, nome_arquivo)
            reader = PdfReader(caminho)
            texto = ""
            for pagina in reader.pages:
                texto += pagina.extract_text() or ""
            textos.append(texto)
    return textos

def dividir_em_trechos(texto, max_tokens=500):
    enc = tiktoken.encoding_for_model("gpt-4")
    palavras = texto.split("\n")
    trechos = []
    atual = ""
    for p in palavras:
        if len(enc.encode(atual + p)) < max_tokens:
            atual += p + "\n"
        else:
            trechos.append(atual.strip())
            atual = p + "\n"
    if atual:
        trechos.append(atual.strip())
    return trechos

def gerar_embeddings_e_salvar():
    textos = carregar_texto_dos_pdfs("documentos")
    trechos = []
    for texto in textos:
        trechos.extend(dividir_em_trechos(texto))

    embeddings = []
    # Enviar em lotes para evitar limite de tokens e acelerar
    for batch in chunks(trechos, 50):
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=batch
        )
        embeddings.extend([e.embedding for e in response.data])

    embeddings_array = np.array(embeddings).astype('float32')

    vetor_db = faiss.IndexFlatL2(1536)
    vetor_db.add(embeddings_array)

    os.makedirs("base_embeddings", exist_ok=True)
    with open("base_embeddings/trechos.pkl", "wb") as f:
        pickle.dump(trechos, f)
    faiss.write_index(vetor_db, "base_embeddings/vetores.index")

def buscar_trechos_relevantes(pergunta):
    with open("base_embeddings/trechos.pkl", "rb") as f:
        trechos = pickle.load(f)
    index = faiss.read_index("base_embeddings/vetores.index")

    emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=[pergunta]
    ).data[0].embedding

    emb_array = np.array([emb]).astype('float32')
    _, indices = index.search(emb_array, k=5)
    selecionados = [trechos[i] for i in indices[0]]
    return "\n\n".join(selecionados)

# Garante um ID único por sessão
@app.before_request
def definir_usuario():
    if "user_id" not in session:
        session["user_id"] = str(uuid4())

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/perguntar', methods=['POST'])
def perguntar():
    data = request.get_json()
    pergunta = data.get('pergunta')
    user_id = session["user_id"]

    if not pergunta:
        return jsonify({'resposta': 'Por favor, digite uma pergunta válida.'}), 400

    contexto = buscar_trechos_relevantes(pergunta)

    mensagens = [
        {"role": "system", "content": "Você é um assistente jurídico. Responda com base no material fornecido."},
        {"role": "user", "content": f"Base de conhecimento:\n{contexto}\n\nPergunta: {pergunta}"}
    ]

    try:
        resposta = client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=mensagens,
            temperature=0.4,
            max_tokens=600
        )
        texto = resposta.choices[0].message.content.strip()
        return jsonify({'resposta': texto})
    except Exception as e:
        print("Erro:", e)
        return jsonify({'resposta': 'Erro ao processar a pergunta.'}), 500

if __name__ == '__main__':
    if not os.path.exists("base_embeddings/vetores.index"):
        gerar_embeddings_e_salvar()
    app.run(debug=True)

