# Kalpit Pvt Ltd, UK – AI Intern Assignments

This repository contains the implementation of Assignment 2: Comprehensive Evaluation Framework for a RAG-based Q&A system on Dr. B.R. Ambedkar's works.

## Overview

This project implements a Retrieval-Augmented Generation (RAG) system for answering questions about Dr. B.R. Ambedkar's writings, along with a comprehensive evaluation framework to assess its performance across multiple metrics and chunking strategies.

## Features

- **RAG Pipeline**: Uses LangChain, ChromaDB, HuggingFace embeddings, and Ollama Mistral 7B
- **Evaluation Metrics**:
  - Retrieval: Hit Rate, Mean Reciprocal Rank (MRR), Precision@K
  - Answer Quality: Relevance, Faithfulness, ROUGE-L Score
  - Semantic: Cosine Similarity, BLEU Score
- **Chunking Strategies**: Small (200-300 chars), Medium (500-600 chars), Large (800-1000 chars)
- **Test Dataset**: 25 pre-defined Q&A pairs covering factual, comparative, conceptual, and unanswerable questions

## Setup Instructions

### Prerequisites

1. **Python 3.8+**: Ensure Python is installed on your system.
2. **Ollama**: Install Ollama and pull the Mistral 7B model:
   ```bash
   curl -fsSL https://ollama.ai/install.sh | sh
   ollama pull mistral
   ```

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/Ravisir21/Kalpit-Pvt-Ltd-UK-Assignment-.git
   cd main
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   

4. Running the Evaluation

To run the comprehensive evaluation:

```bash
python evaluation.py
   ```

## Project Structure

```
AmbedkarGPT-Intern-Task/
├── corpus/                    # Document corpus (6 speech files)
│   ├── speech1.txt
│   ├── speech2.txt
│   ├── speech3.txt
│   ├── speech4.txt
│   ├── speech5.txt
│   └── speech6.txt
├── evaluation.py              # Main evaluation script
├── test_dataset.json          # Test Q&A pairs
├── test_results.json          # Evaluation results (generated)
├── results_analysis.md        # Analysis and recommendations (generated)
├── requirements.txt           # Python dependencies
├── README.md                  # This file
└── TODO.md                    # Task tracking
```

## Running the Evaluation

To run the comprehensive evaluation:

```bash
python evaluation.py
```

This will:
1. Load the document corpus
2. For each chunking strategy (small, medium, large):
   - Create vector embeddings
   - Set up the RAG QA chain
   - Evaluate all 25 test questions
   - Compute all specified metrics
3. Save results to `test_results.json`
4. Generate analysis in `results_analysis.md`

## Expected Output

The evaluation provides answers to:
- Which chunking strategy works best for this corpus?
- What is the system's current accuracy score?
- What are the most common failure types?
- What specific improvements would boost performance?

## Deliverables

- `evaluation.py`: Main evaluation script
- `test_results.json`: Detailed evaluation results
- `corpus/`: Folder containing all 6 documents
- `test_dataset.json`: Test dataset with 25 Q&A pairs
- `results_analysis.md`: Findings and recommendations
- `requirements.txt`: Updated with evaluation dependencies
- `README.md`: Setup and usage instructions

## Technical Stack

- **Python 3.8+**
- **LangChain**: Orchestration framework
- **ChromaDB**: Vector store
- **HuggingFaceEmbeddings**: sentence-transformers/all-MiniLM-L6-v2
- **Ollama**: Mistral 7B LLM
- **Evaluation Libraries**: ragas, rouge-score, nltk, scikit-learn

## Notes

- Ensure Ollama is running with Mistral 7B before executing the evaluation.
- The evaluation may take several minutes to complete, depending on system resources.
- Results are saved in JSON format for easy analysis and visualization.


##### MY RESULT #####

   Look here I had already run this project and i got the final result in results_analysis.md and test_results.json
   You can the output opening this both file.