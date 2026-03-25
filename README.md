# Word Embedding and Name Generation Project

This repository contains code for:
- Corpus collection and preprocessing
- Dataset statistics generation
- Word cloud visualization
- Word embedding experiments
- Name generation models using RNN, BiLSTM, and Attention+RNN

---

## Project Structure

```
project/
│── problem1/
│   ├── corpus.txt
│
│── images/
│   ├── dataset_stats.png
│   ├── wordcloud.png
│
│── src/
│   ├── main.py
│   ├── preprocessing.py
│   ├── train_word2vec.py
│   ├── train_names.py
│
│── requirements.txt
│── README.md
```

---

## Requirements

Install all required packages:

```
pip install -r requirements.txt
```

---

## Running the Code from Terminal

### Clone the repository
```
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### Install dependencies
```
pip install -r requirements.txt
```

### Run the code
```
python main.py
```

---

## Running the Code in Google Colab

### Install dependencies
```
!pip install -r requirements.txt
```

### Run the script
```
!python main.py
```

---

## Dataset Statistics

![Dataset Statistics](images/dataset_stats.png)

---

## Word Cloud

![Word Cloud](images/wordcloud.png)

---

## Notes

- Ensure dataset files are in correct folders
- Update file paths if needed
- GitHub does not allow empty folders

---

## requirements.txt

```
torch>=2.0.0
pypdf>=3.0.0
beautifulsoup4>=4.12.0
wordcloud>=1.9.0
requests>=2.31.0
matplotlib>=3.7.0
lxml>=4.9.0
scikit-learn>=1.3.0
numpy>=1.24.0
pandas>=2.0.0
```

---

## Author

Your Name
