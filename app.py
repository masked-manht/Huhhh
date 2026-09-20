import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

st.set_page_config(page_title="Iz Chat Companion", page_icon="🤖")
st.title("Iz Chat Companion 🤖")

# Chargement du modèle principal (iz-instruct) sur CPU
@st.cache_resource
def load_model():
    model_id = "theplayboy117/iz-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )
    return tokenizer, model

with st.spinner("Chargement de Iz en mémoire..."):
    tokenizer, model = load_model()

# Historique de conversation
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Pose une question à Iz..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Formatage au format ChatML
    full_prompt = "<|im_start|>system\nTu es Iz, un assistant IA intelligent et utile.<|im_end|>\n"
    for m in st.session_state.messages:
        full_prompt += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
    full_prompt += "<|im_start|>assistant\n"

    inputs = tokenizer(full_prompt, return_tensors="pt")

    with st.spinner("Iz réfléchit..."):
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

    response_tokens = outputs[0][inputs.input_ids.shape[-1]:]
    response = tokenizer.decode(response_tokens, skip_special_tokens=True).replace("<|im_end|>", "").strip()

    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
