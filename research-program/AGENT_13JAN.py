from autogen import AssistantAgent

# Setup API key. Add your own API key to config file or environment variable
config_list = [
    {
        "model": "llama3",
        "base_url": "http://localhost:11434/v1",
        'api_key': 'ollama',
    },
]

small = AssistantAgent(name="patient",
                       max_consecutive_auto_reply=2,
                       system_message="You are a patient who has a disease. Tell your symptoms and blood test reports to the doctor.",
                       llm_config={
                           "config_list": config_list,
                           "temperature": 1,
                       })

big = AssistantAgent(name="doctor",
                     max_consecutive_auto_reply=2,
                     system_message="Act as a doctor. Diagonise the patient based on the symptoms and the blood test reports, and give a proper prognosis.",
                     llm_config={
                         "config_list": config_list,
                         "temperature": 1,
                     })

chat_result = big.initiate_chat(small, message="Tell me what are the symptoms you are having, and tell me blood reports. Let me help you. Don't worry, you will be well soon.")

# print(chat_result.chat_history)

# Extracting the conversation
chat_history = chat_result.chat_history

# Open a text file for writing
with open("conversation.txt", "w") as file:
    for entry in chat_history:
        role = entry['name'].capitalize()  # Role as 'Doctor' or 'Patient'
        content = entry['content']        # The actual message
        print(content)
        file.write(f"{role}: {content}\n\n")

print("Conversation has been saved to 'conversation.txt'.")