import flask
import llm
import rag

# Vector Data found at: https://drive.google.com/file/d/12LSAVChAWaD-E_KW71nS51XPsGixUVkT/view?usp=sharing
app = flask.Flask(__name__)

@app.route('/ask', methods=['POST'])
def ask():
    data = flask.request.get_json()
    question = data['question']
    previous_messages = llm.chat_session.history
    rag_data = rag.get_closest_documents(rag.get_vector(question + str(previous_messages)), data["num_documents"], 'vector_data.json')
    similar_info = []
    for item in rag_data:
        similar_info.append(item['document'])
    similar_info = " ".join(similar_info)
    query = f"Question: {question} \n\n Context: {similar_info} \n USE THE CONTEXT TO RESPOND TO THE QUERY OR STATEMENT BUT DONT TALK ABOUT THE CONTEXT"
    print(query)
    response = llm.query(query)

    return flask.jsonify({"response": response, "related_documents": similar_info})

@app.route('/')
def home():
    llm.chat_session = llm.model.start_chat(history=[])
    return flask.render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=5000)