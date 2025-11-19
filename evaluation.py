import json
import os
import numpy as np
import pandas as pd
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters.character import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_classic.chains import RetrievalQA
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
import nltk
nltk.download('punkt')

nltk.download('punkt_tab')


CHUNK_STRATEGIES = {
    'small': {'chunk_size': 250, 'chunk_overlap': 50},
    'medium': {'chunk_size': 550, 'chunk_overlap': 100},
    'large': {'chunk_size': 900, 'chunk_overlap': 150}
}

def load_corpus(corpus_path):
    loader = DirectoryLoader(corpus_path, glob="*.txt", loader_cls=TextLoader)
    documents = loader.load()
    return documents

def create_vectorstore(documents, chunk_strategy):
    text_splitter = CharacterTextSplitter(**chunk_strategy)
    docs = text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(docs, embeddings)
    return vectorstore, docs

def setup_qa_chain(vectorstore):
    llm = Ollama(model="mistral")
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever())
    return qa_chain

def compute_retrieval_metrics(retrieved_docs, relevant_docs, k=5):
    retrieved = [doc.metadata['source'].split('/')[-1] for doc in retrieved_docs[:k]]
    relevant = relevant_docs

    
    hit_rate = 1 if any(doc in retrieved for doc in relevant) else 0

    
    precision_k = len(set(retrieved) & set(relevant)) / k if k > 0 else 0

    
    mrr = 0
    for i, doc in enumerate(retrieved):
        if doc in relevant:
            mrr = 1 / (i + 1)
            break

    return hit_rate, precision_k, mrr

def compute_answer_quality_metrics(answer, ground_truth, retrieved_context):
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    answer_emb = embeddings.embed_query(answer)
    gt_emb = embeddings.embed_query(ground_truth)
    relevance = cosine_similarity([answer_emb], [gt_emb])[0][0]

    
    context_emb = embeddings.embed_query(retrieved_context)
    faithfulness = cosine_similarity([answer_emb], [context_emb])[0][0]

    
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_l = scorer.score(ground_truth, answer)['rougeL'].fmeasure

    return relevance, faithfulness, rouge_l

def compute_semantic_metrics(answer, ground_truth):
    
    reference = nltk.word_tokenize(ground_truth.lower())
    candidate = nltk.word_tokenize(answer.lower())
    bleu = sentence_bleu([reference], candidate)

    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    answer_emb = embeddings.embed_query(answer)
    gt_emb = embeddings.embed_query(ground_truth)
    cos_sim = cosine_similarity([answer_emb], [gt_emb])[0][0]

    return bleu, cos_sim

def evaluate_question(qa_chain, question, ground_truth, source_docs, vectorstore):
    
    result = qa_chain({"query": question})
    answer = result['result']

    
    retrieved_docs = vectorstore.similarity_search(question, k=5)
    retrieved_context = " ".join([doc.page_content for doc in retrieved_docs])

    
    hit_rate, precision_k, mrr = compute_retrieval_metrics(retrieved_docs, source_docs)
    relevance, faithfulness, rouge_l = compute_answer_quality_metrics(answer, ground_truth, retrieved_context)
    bleu, cos_sim = compute_semantic_metrics(answer, ground_truth)

    return {
        'answer': answer,
        'retrieval': {
            'hit_rate': hit_rate,
            'precision@5': precision_k,
            'mrr': mrr
        },
        'answer_quality': {
            'relevance': relevance,
            'faithfulness': faithfulness,
            'rouge_l': rouge_l
        },
        'semantic': {
            'bleu': bleu,
            'cosine_similarity': cos_sim
        }
    }

def main():
    corpus_path = "corpus"
    test_dataset_path = "test_dataset.json"

    
    with open(test_dataset_path, 'r') as f:
        test_data = json.load(f)

    
    documents = load_corpus(corpus_path)

    results = {}

    for strategy_name, chunk_config in CHUNK_STRATEGIES.items():
        print(f"Evaluating {strategy_name} chunking strategy...")

        
        vectorstore, docs = create_vectorstore(documents, chunk_config)
        qa_chain = setup_qa_chain(vectorstore)

        strategy_results = []
        for question_data in test_data['test_questions']:
            q_id = question_data['id']
            question = question_data['question']
            ground_truth = question_data['ground_truth']
            source_docs = question_data['source_documents']

            eval_result = evaluate_question(qa_chain, question, ground_truth, source_docs, vectorstore)
            eval_result['id'] = q_id
            eval_result['question'] = question
            eval_result['ground_truth'] = ground_truth
            eval_result['source_documents'] = source_docs

            strategy_results.append(eval_result)

        results[strategy_name] = strategy_results

        
        vectorstore.delete_collection()

    
    with open('test_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    
    generate_analysis(results)

def generate_analysis(results):
    analysis = "# Evaluation Results Analysis\n\n"

    for strategy, strategy_results in results.items():
        analysis += f"## {strategy.capitalize()} Chunking Strategy\n\n"

        
        retrieval_metrics = {'hit_rate': [], 'precision@5': [], 'mrr': []}
        answer_metrics = {'relevance': [], 'faithfulness': [], 'rouge_l': []}
        semantic_metrics = {'bleu': [], 'cosine_similarity': []}

        for result in strategy_results:
            for key in retrieval_metrics:
                retrieval_metrics[key].append(result['retrieval'][key])
            for key in answer_metrics:
                answer_metrics[key].append(result['answer_quality'][key])
            for key in semantic_metrics:
                semantic_metrics[key].append(result['semantic'][key])

        analysis += "### Retrieval Metrics\n"
        for key, values in retrieval_metrics.items():
            avg = np.mean(values)
            analysis += f"- {key}: {avg:.3f}\n"

        analysis += "\n### Answer Quality Metrics\n"
        for key, values in answer_metrics.items():
            avg = np.mean(values)
            analysis += f"- {key}: {avg:.3f}\n"

        analysis += "\n### Semantic Metrics\n"
        for key, values in semantic_metrics.items():
            avg = np.mean(values)
            analysis += f"- {key}: {avg:.3f}\n"

        analysis += "\n"

    # Overall recommendations
    analysis += "## Recommendations\n\n"
    # Compare strategies and provide insights based on metrics

    with open('results_analysis.md', 'w') as f:
        f.write(analysis)

if __name__ == "__main__":
    main()
