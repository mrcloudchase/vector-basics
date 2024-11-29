# vector-basics

Getting started with vector search using OpenAI without a vector database.

## Setup

1. Install the requirements

```bash
pip install -r requirements.txt
```

2. Set the OpenAI API key in the `.env` file

```bash
OPENAI_API_KEY=<your-openai-api-key>
```

## Run the script

```bash
python vector_search.py
```

## Describe the script

The script generates embeddings for a list of sentences and then allows a user to query the embeddings using a natural language question. It then uses cosine similarity to find the top N most similar sentences to the query and passes them to a LLM to answer the question.


### Code Explanation

1. `generate_embedding`

This function generates an embedding for a given text using the OpenAI API.

2. `calculate_cosine_similarity`

This function calculates the cosine similarity between two vectors.

3. `get_top_n_similar`

This function finds the top N most similar sentences to a given query using cosine similarity.

4. `rag_gpt4o`

This function uses the RAG technique to answer a user's question using the top N most similar sentences.

## Explaination of what a vector is

A vector is a representation of a sentence in a high dimensional space. It is a list of numbers that represent the sentence. The cosine similarity between two vectors is a measure of how similar the two sentences are. It is calculated by taking the dot product of the two vectors and dividing by the product of the magnitudes of the vectors.

## Explaination of cosine similarity

Cosine similarity is a measure of how similar two vectors are. It is calculated by taking the dot product of the two vectors and dividing by the product of the magnitudes of the vectors.

## Explaination of RAG

RAG is a technique for answering questions using a large language model (LLM) and a vector database. The LLM is used to answer the question and the vector database is used to find the most similar sentences to the question. The most similar sentences are then passed to the LLM to answer the question.

We aren't using a vector database in this example, but the process is the same. We are using the cosine similarity function to find the most similar sentences to the question.

## Why should we use a vector database?

Vector databases are optimized for similarity search, which makes them a good fit for RAG systems. They are also designed to handle the high dimensional data that embeddings produce.

Without using a vector database, we would have to search through all the sentences every time we wanted to answer a question. This would be very slow and inefficient. A vector database allows us to efficiently search for the most similar sentences to a given query.