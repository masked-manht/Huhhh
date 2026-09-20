import streamlit as st
from huggingface_hub import InferenceClient

st.set_page_config(page_title="Iz Chat Companion", page_icon="🤖")
st.title("Iz Chat Companion 🤖")

MODEL_ID = "theplayboy117/iz-instruct"
HF_TOKEN = "Hf_zEsdWILqXGPoFtjEoYxvpChQrNtBohwYMI"

client = InferenceClient(MODEL_ID, token=HF_TOKEN)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Pose une question à Iz..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Reconstruction du prompt au format ChatML
    full_prompt = "<|im_start|>system\nTu es Iz, un assistant IA intelligent et utile.<|im_end|>\n"
    for m in st.session_state.messages:
        full_prompt += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
    full_prompt += "<|im_start|>assistant\n"

    with st.spinner("Iz réfléchit..."):
        try:
            response = client.text_generation(
                full_prompt,
                max_new_tokens=250,
                temperature=0.7,
                return_full_text=False
            )
            clean_response = response.replace("<|im_end|>", "").strip()
        except Exception as e:
            clean_response = f"⚠️ Le modèle est en cours de réveil ou indisponible : {e}"

    with st.chat_message("assistant"):
        st.markdown(clean_response)
    st.session_state.messages.append({"role": "assistant", "content": clean_response})
    
