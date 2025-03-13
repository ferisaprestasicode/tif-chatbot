from openai import OpenAI
from tools.error_log import TopErrorDevices, TopErrorCategories, CountErrors
from tools.monitoring_anomalies import AnomaliesCount, AnomaliesMonitoring, TrafficForecasting
from tools.traffic_forecasting import TrafficChanges, TrafficPredict
import json
import streamlit as st
import logging
from prompt.system import get_system_prompt

logger = logging.getLogger('TIF')

# Run conversation function
def run_conversation(user_prompt: str, system_prompt: str, conversation_history):
    print("Starting new conversation")
    logger.debug(f"User prompt: {user_prompt}")
    
    try:
        # MODEL = "gemini-2.0-flash"

        # # OpenAI Client
        # client = OpenAI(
        #     api_key="AIzaSyCG-GBc7Q8GmUlS9mFTavyXA64lv8E4Zkc",
        #     base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        # )

        MODEL = 'gpt-3.5-turbo'
        client = OpenAI()
        print(f"Using model: {MODEL}")
        
        messages = [
            {"role": "system", "content": system_prompt},
        ] + conversation_history + [
            {"role": "user", "content": user_prompt},
        ]

        # List tools
        tools = [TopErrorDevices(), TopErrorCategories(), CountErrors(),
                 AnomaliesCount(),AnomaliesMonitoring(), TrafficForecasting(), 
                 TrafficPredict(),TrafficChanges()]
        
        tools_scheme = []
        for tool in tools:
            tool_schema = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.args_schema.schema()
                }
            }
            tools_scheme.append(tool_schema)
        print(f"Initialized {len(tools)} tools")

        logger.debug("Sending initial request to model")
        response = client.chat.completions.create(
            model=MODEL,
            temperature=0.3,
            messages=messages,
            tools=tools_scheme,
            max_tokens=4096
        )
        print(f"Response : {response} ")

        

        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls
        print(f"Received response with {len(tool_calls) if tool_calls else 0} tool calls")

        if not tool_calls:
            print("No tool calls required, returning direct response")
            return {"response": response_message.content, "tools": None,  "cs_agent": False}

        final_response = None
        final_tools_response = None
        cs_agent = False

        while tool_calls:
            available_functions = {
                "count_errors": CountErrors().run,
                "top_error_devices": TopErrorDevices().run,
                "top_error_categories": TopErrorCategories().run,
                "anomalies": AnomaliesMonitoring().run,
                "hitung_device_anomalies": AnomaliesCount().run,
                "traffic_forecasting": TrafficForecasting().run,
                "traffic_predict": TrafficPredict().run,
                "traffic_changes": TrafficChanges().run
            }
            messages.append(response_message)

            for tool_call in tool_calls:
                function_name = tool_call.function.name
                print(f"Executing tool: {function_name}")
                
                function_to_call = available_functions[function_name]
                function_args = json.loads(tool_call.function.arguments)
                logger.debug(f"Tool arguments: {function_args}")
                
                function_response = function_to_call(**function_args)
                logger.debug(f"Tool response: {function_response}")

                messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": json.dumps(function_response),
                })
                final_tools_response = function_response
                if isinstance(final_tools_response, dict) and final_tools_response.get("tools"):
                    cs_agent = True


            logger.debug("Sending follow-up request to model")
            print(f"ISI TOOLS: {function_response}")
            next_response = client.chat.completions.create(
                model=MODEL,
                temperature=0.3,
                messages=messages,
                tools=tools_scheme
            )
            response_message = next_response.choices[0].message
            tool_calls = response_message.tool_calls
            
            if not tool_calls:
                final_response = response_message.content
                print("Conversation complete")
                break
                    
        return {"response": final_response, "tools": final_tools_response, "cs_agent": cs_agent}
    
    except Exception as e:
        print(e)
        logger.error(f"Error in run_conversation: {str(e)}", exc_info=True)
        return {
            "response": "Maaf, terjadi kesalahan internal. Silakan coba lagi nanti.",
            "tools": None,
            "cs_agent": False
        }
    

def main():
    print("Starting TIF Network Error Assistant")
    st.title("TIF Network Assistant")
    

    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
        print("Initialized new conversation history")

    system_prompt = get_system_prompt()

    # Display conversation history
    for message in st.session_state.conversation_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # User input
    user_input = st.chat_input("Masukkan pesan Anda:")

    if user_input:
        print("Received new user input")
        
        with st.chat_message("user"):
            st.write(user_input)

        st.session_state.conversation_history.append({"role": "user", "content": user_input})
        
        try:
            response = run_conversation(user_input, system_prompt, st.session_state.conversation_history)
            
            if response['response'] == "Maaf, terjadi kesalahan internal. Silakan coba lagi nanti.":
                logger.error("Internal error occurred during conversation")
                st.error(response['response'])
            else:
                print("Successfully processed user input")
                with st.chat_message("assistant"):
                    st.write(response['response'])
                st.session_state.conversation_history.append({"role": "assistant", "content": response['response']})
        
        except Exception as e:
            logger.error(f"Error processing user input: {str(e)}", exc_info=True)
            st.error("Terjadi kesalahan saat memproses permintaan Anda. Silakan coba lagi nanti.")

if __name__ == "__main__":
    main()