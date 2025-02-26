from flask import Blueprint, jsonify, request, render_template, session, redirect, url_for, flash
import logging
from datetime import datetime
import json
import uuid
import openai
from google.auth import default, transport
import vertexai
from dotenv import load_dotenv

from config import project_id
from tools.error_log import TelkomAnalyzer
from tools.monitoring_anomalies import AnomaliesMonitoring
from tools.traffic_forecasting import TrafficForecasting
from prompt.system import get_system_prompt
from db.routes import insert_history_chat, fetch_chat_history_by_email, delete_session_chat, save_message, fetch_conversation_history
from config import logger

# Initialize Blueprint
chat_gen_bp = Blueprint('chat_gen', __name__, template_folder='templates', static_folder='static')

# Initialize Vertex AI
location = "us-central1"
vertexai.init(project=project_id, location=location)

# Get credentials
credentials, _ = default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
auth_request = transport.requests.Request()
credentials.refresh(auth_request)

def run_conversation(user_prompt: str, conversation_history):
    logger.info("Starting new conversation")
    logger.debug(f"User prompt: {user_prompt}")
    
    try:
        MODEL = "google/gemini-1.5-flash"
        client = openai.OpenAI(
            base_url=f"https://{location}-aiplatform.googleapis.com/v1beta1/projects/{project_id}/locations/{location}/endpoints/openapi",
            api_key=credentials.token,
        )

        logger.info(f"Using model: {MODEL}")
        
        messages = conversation_history + [
            {"role": "user", "content": user_prompt},
        ]

        # List tools
        tools = [TelkomAnalyzer(), AnomaliesMonitoring(), TrafficForecasting()]
        
        tools_scheme = []
        for tool in tools:
            tool_schema = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            key: {
                                "type": prop.get('type', 'string'),
                                "description": prop.get('description', '')
                            } for key, prop in tool.args_schema.model_json_schema().get('properties', {}).items()
                        },
                        "required": tool.args_schema.model_json_schema().get('required', [])
                    }
                }
            }
            tools_scheme.append(tool_schema)
        logger.info(f"Initialized {len(tools)} tools")

        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools_scheme
        )
        
        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls

        if not tool_calls:
            return response_message.content

        final_response = None
        while tool_calls:
            available_functions = {
                "error": TelkomAnalyzer().run,
                "anomalies": AnomaliesMonitoring().run,
                "traffic_forecasting": TrafficForecasting().run  
            }
            messages.append(response_message)

            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_to_call = available_functions[function_name]
                function_args = json.loads(tool_call.function.arguments)
                
                function_response = function_to_call(**function_args)
                messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": json.dumps(function_response),
                })

            next_response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=tools_scheme
            )
            response_message = next_response.choices[0].message
            tool_calls = response_message.tool_calls
            
            if not tool_calls:
                final_response = response_message.content
                break
                    
        return final_response
    
    except Exception as e:
        logger.error(f"Error in run_conversation: {str(e)}", exc_info=True)
        return "Maaf, terjadi kesalahan internal. Silakan coba lagi nanti."

@chat_gen_bp.route('/delete-session/<string:session_id>', methods=['DELETE'])
def delete_session(session_id):
    try:
        delete_session_chat(session_id)
        return jsonify({"message": f"Success delete data: {session_id}"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@chat_gen_bp.route('/chat', methods=['POST'])
def chat():
    user_message = request.json['message']
    session_account =request.json['session_account'].strip()
    session_id = request.json['session_id']
    if not session_id:
        session_id = str(uuid.uuid4())
    
    try:
        chat_history = fetch_conversation_history(session_id)
        
        # Generate response
        response = run_conversation(user_message, chat_history)
        
        # Create response object
        response_obj = {
            'text': response,
            'response_id': str(uuid.uuid4()),
            'timestamp': datetime.now().isoformat()
        }
        
        save_message(session_id, "user", user_message)
        save_message(session_id, "assistant", response)
        insert_history_chat(session_id, session_account, user_message, response, category="general")

        return jsonify({
            'response': response_obj,
            'session_id': session_id,
            'session_account': session_account
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@chat_gen_bp.route('/feedback', methods=['POST'])
def feedback():
    try:
        data = request.json
        print(f"Received feedback: {data}")
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500