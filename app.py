import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import os
import re

# -------------------------------------------------------------
# PyTorch Model & Tokenizer Definitions
# -------------------------------------------------------------

class LSTMWordPredictor(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(LSTMWordPredictor, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hn, cn) = self.lstm(embedded)
        last_timestep_out = lstm_out[:, -1, :]
        logits = self.fc(last_timestep_out)
        return logits

class SimpleTokenizer:
    def __init__(self):
        self.word_index = {}
        self.index_word = {}
        self.total_words = 0

    def fit_from_dict(self, word_index, index_word):
        self.word_index = word_index
        self.index_word = {int(k): v for k, v in index_word.items()}
        self.total_words = len(self.word_index) + 1

    def texts_to_sequences(self, texts):
        sequences = []
        for t in texts:
            words = re.findall(r"\b\w+\b", t)
            seq = [self.word_index[word] for word in words if word in self.word_index]
            sequences.append(seq)
        return sequences

# -------------------------------------------------------------
# Configuration & CSS Styling (Premium Theme)
# -------------------------------------------------------------

st.set_page_config(
    page_title="Shakespeare Next-Word Predictor",
    page_icon="📖",
    layout="centered"
)

# Custom premium dark mode and glassmorphism styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
    font-family: 'Outfit', sans-serif;
    background-color: #090D16;
    color: #F1F5F9;
}

/* Glassmorphism sidebar */
[data-testid="stSidebar"] {
    background-color: #0E1322 !important;
    border-right: 1px solid rgba(255, 255, 255, 0.05);
}

/* Titles */
.gradient-title {
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(135deg, #38BDF8 10%, #818CF8 50%, #C084FC 90%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 0.2rem;
    filter: drop-shadow(0 2px 10px rgba(129, 140, 248, 0.25));
}

.subtitle-desc {
    font-size: 1.15rem;
    color: #94A3B8;
    text-align: center;
    margin-bottom: 2rem;
}

/* Modern inputs & buttons */
.stTextInput > div > div > input {
    background-color: rgba(15, 23, 42, 0.6) !important;
    color: #F8FAFC !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 8px !important;
    padding: 12px 16px !important;
    font-size: 1.1rem;
    transition: all 0.3s ease !important;
}

.stTextInput > div > div > input:focus {
    border-color: #818CF8 !important;
    box-shadow: 0 0 0 3px rgba(129, 140, 248, 0.2) !important;
}

/* Cards & containers */
.glass-panel {
    background: rgba(17, 25, 40, 0.7);
    backdrop-filter: blur(12px);
    border-radius: 12px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    padding: 24px;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
    margin-bottom: 20px;
}

.prediction-box {
    background: rgba(15, 23, 42, 0.8);
    border-radius: 8px;
    border: 1px solid rgba(129, 140, 248, 0.2);
    padding: 20px;
    min-height: 80px;
    font-size: 1.3rem;
    line-height: 1.6;
    margin-top: 15px;
    margin-bottom: 20px;
}

.highlight-gen {
    background: linear-gradient(135deg, #38BDF8, #818CF8, #C084FC);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# Main Application Header
# -------------------------------------------------------------

st.markdown('<h1 class="gradient-title">📖 Shakespeare Next-Word Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle-desc">Generative PyTorch LSTM model trained on the classic corpus of Hamlet</p>', unsafe_allow_html=True)

MODEL_PATH = "next_word_lstm.pth"

if not os.path.exists(MODEL_PATH):
    st.error("❌ Model Checkpoint File Not Found!")
    st.write(
        "Please train the model first by running the `train_pytorch.ipynb` notebook to generate the `next_word_lstm.pth` file."
    )
else:
    # -------------------------------------------------------------
    # Predictor Interface (Model is loaded)
    # -------------------------------------------------------------
    
    @st.cache_resource
    def load_cached_model():
        checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Build tokenizer
        tokenizer = SimpleTokenizer()
        tokenizer.fit_from_dict(checkpoint['word_index'], checkpoint['index_word'])
        
        # Load model architecture
        model = LSTMWordPredictor(
            vocab_size=tokenizer.total_words,
            embedding_dim=100,
            hidden_dim=150
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        return model, tokenizer, checkpoint['max_sequence_len']

    model, tokenizer, max_sequence_len = load_cached_model()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Generator Parameters in Sidebar
    st.sidebar.markdown('<h3 style="color: #818CF8; margin-top: 0;">🎛️ Generator Controls</h3>', unsafe_allow_html=True)
    num_words_to_predict = st.sidebar.slider("Words to Generate", min_value=1, max_value=25, value=5)
    temperature = st.sidebar.slider(
        "Temperature (Creativity)", 
        min_value=0.0, 
        max_value=2.0, 
        value=0.7, 
        step=0.1,
        help="0.0 makes the model deterministic (most probable word). Higher values introduce randomness/creativity."
    )

    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown("<h3 style='margin-top: 0; color: #F1F5F9;'>📜 Enter Seed Text</h3>", unsafe_allow_html=True)
    st.write("Type a phrase (e.g. *'to be or not to'*, *'the tragedie of'*, *'speak to me'*) and see the PyTorch LSTM generate the continuation:")
    
    seed_phrase = st.text_input("Seed Input:", value="To be or not to", key="seed_input")
    
    # Inference Loop
    def perform_inference(seed, count, temp):
        current = seed
        generated = []
        first_probabilities = None
        
        for step in range(count):
            token_list = tokenizer.texts_to_sequences([current])[0]
            seq_len = max_sequence_len - 1
            pad_len = seq_len - len(token_list)
            
            if pad_len > 0:
                token_list = [0] * pad_len + token_list
            else:
                token_list = token_list[-seq_len:]
                
            input_tensor = torch.tensor([token_list], dtype=torch.long).to(device)
            
            with torch.no_grad():
                outputs = model(input_tensor)
                logits = outputs[0]
                
                if temp == 0.0:
                    probs = torch.softmax(logits, dim=0)
                    predicted_idx = torch.argmax(logits).item()
                else:
                    scaled_logits = logits / temp
                    probs = torch.softmax(scaled_logits, dim=0)
                    predicted_idx = torch.multinomial(probs, num_samples=1).item()
                    
                # Save prediction details for the first step
                if step == 0:
                    actual_probs = torch.softmax(logits, dim=0)
                    top5_probs, top5_indices = torch.topk(actual_probs, 5)
                    first_probabilities = []
                    for p, idx in zip(top5_probs.tolist(), top5_indices.tolist()):
                        word = tokenizer.index_word.get(idx, f"<unknown:{idx}>")
                        first_probabilities.append((word, p))
                        
            output_word = tokenizer.index_word.get(predicted_idx, "")
            if not output_word:
                break
            current += " " + output_word
            generated.append(output_word)
            
        return generated, first_probabilities

    if seed_phrase.strip() == "":
        st.warning("Please type a seed phrase to generate words.")
    else:
        generated_words, next_word_probs = perform_inference(seed_phrase, num_words_to_predict, temperature)
        
        # Print text result
        st.markdown("#### Generated Output:")
        generated_str = " ".join(generated_words)
        st.markdown(f"""
        <div class="prediction-box">
            {seed_phrase} <span class="highlight-gen">{generated_str}</span>
        </div>
        """, unsafe_allow_html=True)
        
        # Show Probability Analysis Chart
        if next_word_probs:
            st.markdown("#### 📊 Top 5 Predictions for the Next Word:")
            st.write("Here are the probabilities calculated by the model's output layer for the first generated word:")
            
            for word, prob in next_word_probs:
                percent = prob * 100
                st.markdown(f"""
                <div style="margin-bottom: 14px;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 3px;">
                        <span style="font-weight: 600; color: #E2E8F0; font-size: 1rem;">"{word}"</span>
                        <span style="color: #38BDF8; font-weight: 600; font-size: 0.95rem;">{percent:.2f}%</span>
                    </div>
                    <div style="background-color: #1E293B; border-radius: 4px; height: 10px; width: 100%;">
                        <div style="background: linear-gradient(90deg, #3B82F6 0%, #8B5CF6 100%); height: 10px; border-radius: 4px; width: {percent}%;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
