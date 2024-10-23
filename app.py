from flask import Flask, render_template, request, jsonify, url_for
import education_chatbot
import education_chatbot_full as cpg

app = Flask(__name__)

# Function to get response from the Education_ChatBot model
def get_response_from_chatbot(user_message):
    # Adjust the source paths to point to the static directory with forward slashes
    answer, source = cpg.Education_ChatBot(user_message)
    
    # Use forward slashes for URLs
    source = [url_for('static', filename=f'TestImages/{img}') for img in source]
    return answer, source

@app.route('/')
def chat():
    return render_template('chat.html')

@app.route('/send_message', methods=['POST'])
def send_message():
    user_message = request.json.get("message")
    bot_response, sources = get_response_from_chatbot(user_message)
    print(sources)

    # Return the response along with the sources
    return jsonify({"response": bot_response, "sources": sources})

if __name__ == '__main__':
    app.run(debug=True)
