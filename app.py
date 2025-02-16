
import streamlit as st
from transformers import pipeline

# Load the models
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")
generation_pipeline = pipeline("text2text-generation", model="google/flan-t5-large")

# Define the chatbot response logic
def healthcare_chatbot(user_input):
    # Basic rule-based responses
    if "symptom" in user_input:
        return "It seems you're experiencing symptoms. Please consult a healthcare provider for a proper diagnosis."
    elif "appointment" in user_input:
        return "Would you like assistance in booking an appointment with a doctor?"
    elif "medication" in user_input:
        return "It's essential to take medications as prescribed. If you have concerns, consult your doctor or pharmacist."
    else:
        # Factual response using QA model
        try:
            context = (
                "Healthcare involves symptoms, medications, treatments, and appointments. "
                "It's essential to consult a doctor for accurate advice."
            )
            qa_response = qa_pipeline(question=user_input, context=context)
            if qa_response['score'] > 0.3:
                return qa_response['answer']
        except Exception as e:
            print("Error with QA model:", e)

        # Fallback to natural language generation
        try:
            response = generation_pipeline(
                f"Provide a helpful healthcare response for the query: {user_input}", 
                max_length=150, 
                num_return_sequences=1
            )
            return response[0]['generated_text']
        except Exception as e:
            print("Error with NLG model:", e)
            return "I'm sorry, I couldn't process your query. Please consult a healthcare professional."

# Streamlit web app interface
def main():
    st.title("Healthcare Assistant Chatbot")

    user_input = st.text_input("How can I assist you today?", "")
    
    if st.button("Submit"):
        if user_input:
            st.write("User: ", user_input)
            response = healthcare_chatbot(user_input)
            st.write("Healthcare Assistant: ", response)
        else:
            st.write("Please enter a query.")

if __name__ == "__main__":
    main()
