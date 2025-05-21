from flask import Flask, request, jsonify, render_template, session
from flask_cors import CORS
from openai import OpenAI
from dotenv import load_dotenv
from uuid import uuid4
import os

# Carregar variáveis de ambiente
load_dotenv()

# Inicialização do app
app = Flask(__name__, static_folder='static', template_folder='templates')
app.secret_key = os.getenv("SECRET_KEY", "chave_secreta_padrao")  # necessário para sessões
CORS(app)

# Cliente OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Dicionário que guarda o histórico por usuário
conversas = {}

@app.before_request
def definir_usuario():
    """Garante que cada usuário tenha um ID único para sessão"""
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

    # Iniciar histórico se for a primeira pergunta do usuário
    if user_id not in conversas:
        conversas[user_id] = [
            {"role": "system", "content": "Você é um assistente jurídico acadêmico. Ajude estudantes de Direito a entender leis, doutrinas e conceitos jurídicos, especialmente do Direito Brasileiro."}
        ]

    # Adiciona a pergunta do usuário ao histórico
    conversas[user_id].append({"role": "user", "content": pergunta})

    try:
        resposta = client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=conversas[user_id],
            temperature=0.4,
            max_tokens=600
        )
        texto = resposta.choices[0].message.content.strip()

        # Adiciona a resposta do bot ao histórico
        conversas[user_id].append({"role": "assistant", "content": texto})

        return jsonify({'resposta': texto})

    except Exception as e:
        print("Erro:", e)
        return jsonify({'resposta': 'Erro ao processar a pergunta.'}), 500

if __name__ == '__main__':
    app.run(debug=True)

