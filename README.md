# Shakespearean Next-Word Prediction using PyTorch LSTM

A professional, end-to-end Deep Learning system that learns language sequences from Shakespeare's *Hamlet* and predicts upcoming words in real-time. The project features a robust **PyTorch** training pipeline utilizing GPU acceleration (CUDA) and an interactive **Streamlit** web application for text generation.

---

## 🚀 Key Highlights

*   **PyTorch Framework:** Built natively using PyTorch, showcasing modern class-based model definitions, custom dataset abstractions, and optimized tensor execution.
*   **LSTM Recurrent Architecture:** Implements a Long Short-Term Memory (LSTM) network to effectively capture long-range contextual associations and semantic dependencies in sequential text data.
*   **Hardware Acceleration (NVIDIA CUDA):** Fully configured to execute training and real-time inference on dedicated GPUs (e.g., NVIDIA GeForce RTX 4050), accelerating training runs to under a minute.
*   **Interactive Web Workspace:** A premium Streamlit dashboard incorporating custom dark-theme glassmorphism CSS, real-time generation text highlighting, and Top-5 next-word probability distributions.

---

## 🧠 Core Architecture & Theory

### Why LSTM (Long Short-Term Memory)?
Recurrent Neural Networks (RNNs) struggle with long-term text dependencies due to the **vanishing gradient problem**. LSTMs resolve this by introducing a **Cell State** regulated by three specialized gates:
1.  **Forget Gate:** Decides what information from previous words should be discarded.
2.  **Input Gate:** Identifies which new semantic information from the current word should be stored in the cell state.
3.  **Output Gate:** Determines the next hidden state (and subsequent word prediction) based on the updated cell state.

This makes LSTMs exceptionally suited for next-word prediction, where the network must remember grammatical structure and context (like subjects and verb agreements) over several preceding tokens.

### PyTorch Network Setup
The network is structured as follows:
```python
LSTMWordPredictor(
  (embedding): Embedding(vocab_size, embedding_dim=100, padding_idx=0)
  (lstm): LSTM(embedding_dim=100, hidden_dim=150, batch_first=True)
  (fc): Linear(in_features=150, out_features=vocab_size)
)
```
*   **Embedding Layer:** Maps integer token IDs to dense vectors of size `100`, capturing semantic similarities between words.
*   **LSTM Layer:** Processes the embedded token sequence through a `150`-dimensional hidden state.
*   **Fully Connected (FC) Layer:** Projects the final step output of the LSTM back into the vocabulary space to produce raw logits for all possible words.

---

## 📁 Project Structure

*   `train_pytorch.ipynb` - Jupyter notebook containing CUDA diagnostics, raw text preprocessing, model definition, training loop, and checkpoint saving.
*   `app.py` - Standard Streamlit web application showcasing inference inputs, generation parameters, and next-word visual analysis.
*   `hamlet.txt` - Shakespeare's complete *Tragedy of Hamlet*, containing the raw text corpus used to fit the vocabulary.
*   `next_word_lstm.pth` - Saved model weights, vocabulary dictionaries, and sequence metadata checkpoint.
*   `requirements.txt` - Project dependencies.

---

## 🛠️ Setup & Installation

### 1. Set Up Virtual Environment
Create and activate a isolated Python virtual environment:
```bash
# Create myvenv
python -m venv myvenv

# Activate myvenv (Windows PowerShell)
.\myvenv\Scripts\Activate.ps1
```

### 2. Install Dependencies
To install standard packages along with **CUDA GPU support** for PyTorch (highly recommended if you have an NVIDIA GPU):
```bash
pip install torch torchinfo --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### 3. Model Training
Run all cells in [train_pytorch.ipynb](file:///c:/Users/Athul%20VR/OneDrive/Desktop/Word%20Prediction%20using%20LSTM/train_pytorch.ipynb) to train the model. This will fit the custom tokenizer, compile the training sequences, and output the model checkpoint (`next_word_lstm.pth`).

### 4. Launch the Streamlit Web Application
Execute the following command to start the local web app:
```bash
streamlit run app.py
```
Open **[http://localhost:8501](http://localhost:8501)** in your browser to interact with the model.

---

## 🔮 Generative Features

*   **Temperature-Scaled Inference:** The UI features a **Temperature (Creativity)** slider. 
    *   A temperature of `0.0` uses deterministic *greedy decoding* (always choosing the single highest probability word).
    *   Temperatures from `0.1` to `2.0` scale logits before running a multinomial random sample, allowing for more creative and poetic Shakespearean phrasings.
*   **Top 5 Probability Chart:** Displays a live, animated horizontal bar chart visualizing the top 5 candidates the LSTM is considering for the next token, showing its raw confidence level for each word.
