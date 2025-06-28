
# 🧠 News Summarization using LSTM (Deep Learning Approach)

This project focuses on **extractive and abstractive text summarization** of news articles using deep learning techniques, specifically an **LSTM-based Sequence-to-Sequence (Seq2Seq) model**. The objective is to generate concise summaries from long-form news content while preserving the essential information.


## 📌 Features

- Preprocesses and summarizes real-world news datasets
- Converts text into numerical vectors using word embeddings
- Implements a Seq2Seq LSTM model (with optional attention mechanism)
- Performs both extractive and abstractive summarization
- Evaluates the summaries using **BLEU** and **ROUGE** scores


## 🗃️ Dataset

Two datasets are used:

- `news_summary.csv` — Smaller dataset (~4,500 entries)
- `news_summary_more.csv` — Larger dataset (~10,000 entries)

Each dataset contains:

- `headlines`: Title of the news article
- `text`: Main body of the news
- Optional: `author`, `date`


## 🔧 Installation & Requirements

Install the required Python libraries:

```bash
pip install numpy pandas tensorflow keras scikit-learn matplotlib seaborn
```

If using **Google Colab**, mount your Google Drive as shown:

```python
from google.colab import drive
drive.mount('/content/drive')
```


## 🚀 How to Run

Follow these steps to run the project:

1. **Clone this repository** or upload the notebook to [Google Colab](https://colab.research.google.com/).
2. **Mount Google Drive** to access the dataset files:

   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

3. **Run the notebook cells sequentially**, in this order:

   - ✅ Check system and CPU/GPU info
   - 📊 Load and inspect the dataset
   - 🧹 Preprocess text (cleaning, tokenization, stemming)
   - 🔠 Apply word embedding using `Tokenizer` and padding
   - 🧠 Build the LSTM Seq2Seq model
   - 📈 Train and evaluate the model
   - 📝 View the generated summaries and performance metrics


## 🧠 Model Architecture

The summarization model uses an **LSTM-based Sequence-to-Sequence architecture**:

- **Encoder**:
  - Embedding Layer
  - LSTM Layer
- **Decoder**:
  - Embedding Layer
  - LSTM Layer
  - Dense Output Layer
- 🔁 *Optional*: Add an Attention Mechanism for better context
- 🧑‍🏫 Training Strategy: *Teacher Forcing* on padded sequences


## 📈 Results

- Trained for multiple epochs for better accuracy
- Summaries were more coherent and context-aware with larger datasets
- Evaluation metrics:
  - **BLEU Score**
  - **ROUGE Score**
- Experiments conducted:
  - With 1,000 rows only
  - With 2 LSTM layers
  - With the entire dataset using a single LSTM

**Example Output**:

> **Original**: The government has launched a new initiative to improve rural healthcare...  
>  
> **Generated Summary**: Government launches rural healthcare initiative.


## ⚠️ Limitations & Future Work

- Training on larger datasets (e.g., **CNN/DailyMail**) could improve generalization
- Replacing LSTM with **transformer-based models** (e.g., BERT, T5) may improve performance
- Explore **API integration** or **web-based UI** for user-friendly deployment


## 📄 License

This project is licensed under the [MIT License](LICENSE).


## ✍️ Credits

- **Notebook Author**: *Meghana Naidu T*
- **Guided By**: *Dr. Rajeev Sharma*
- **Dataset Source**: Public repositories on *GitHub*
