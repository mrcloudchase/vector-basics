from openai import OpenAI # Import the OpenAI client
from dotenv import load_dotenv # Import the dotenv library
import os # Import the os library
import numpy as np # Import the numpy library
import matplotlib.pyplot as plt # Import the matplotlib library


# Load the API key from the .env file
load_dotenv()

# Initialize the OpenAI client
client = OpenAI(
    # Set the API key
    api_key=os.getenv("OPENAI_API_KEY"),
)

# List of sentences to be embedded
sentences = [
    # Restaurants by Cuisine
    "Best Italian restaurants in downtown Manhattan.",
    "Authentic Mexican taquerias in Brooklyn.",
    "Top-rated sushi bars in Los Angeles.",
    "Famous Chinese dim sum places in San Francisco.",
    "Indian curry houses with great reviews.",
    
    # Price Points
    "Luxury fine dining experiences under $200.",
    "Budget-friendly ethnic restaurants nearby.",
    "Mid-range family restaurants with good portions.",
    "Affordable lunch spots for office workers.",
    "High-end tasting menu restaurants.",
    
    # Specific Food Items
    "Best wood-fired Neapolitan pizza places.",
    "Restaurants known for fresh handmade pasta.",
    "Places serving authentic ramen bowls.",
    "Restaurants with award-winning burgers.",
    "Best places for fresh seafood platters.",
    
    # Dietary Preferences
    "Top-rated vegan restaurants in the city.",
    "Gluten-free friendly bakeries and cafes.",
    "Restaurants with extensive vegetarian options.",
    "Organic farm-to-table restaurants.",
    "Keto-friendly dining options.",
    
    # Quality/Reviews
    "Restaurants with Michelin stars.",
    "Poorly rated restaurants to avoid.",
    "Restaurants with health code violations.",
    "Most complained about dining spots.",
    "Overpriced restaurants with bad service."
]

# Function to generate the embedding of text passed in
def generate_embedding(text, model="text-embedding-3-small"):
    # Create the embedding of the text using the OpenAI client
    embedding = client.embeddings.create(input=[text], model=model).data[0].embedding
    # Return the embedding for use in other functions at global scope
    return embedding

# Create a dictionary to store the sentence vectors
sentence_vectors = {}

# Loop through the sentences and create the embedding
for sentence in sentences:
    # Generate the embedding of the sentence in the sentences list
    embedding = generate_embedding(sentence)
    # Store the embedding in the sentence_vectors dictionary
    sentence_vectors[sentence] = embedding

# Function to calculate cosine similarity
def calculate_cosine_similarity(query_vector, vector):
    # Calculate the cosine similarity between the query vector and the sentence vector by taking the dot product and dividing by the product of the magnitudes of the vectors
    return np.dot(query_vector, vector) / (np.linalg.norm(query_vector) * np.linalg.norm(vector))

# Function to get the top n similar sentences
def get_top_n_similar(query_sentence, n=2):
    query_embedding = generate_embedding(query_sentence)
    similarities = {sentence: calculate_cosine_similarity(query_embedding, sentence_vectors[sentence]) for sentence in sentences}
    sorted_similarities = dict(sorted(similarities.items(), key=lambda x: x[1], reverse=True))
    top_matches = list(sorted_similarities.items())[:n]
    # Print the top n similar sentences
    for sentence, score in top_matches:
        print(f"Similarity: {score:.4f} - {sentence}")
    return top_matches

# Function for RAG using GPT-4o
def rag_gpt4o(query_sentence, n=2):
    # Get the top n similar sentences
    top_matches = get_top_n_similar(query_sentence, n)
    
    # Create a better formatted context from the matches
    context = "\n".join([f"- {sentence}" for sentence, _ in top_matches])
    
    # Use GPT-4o with a better system prompt and user question
    response = client.chat.completions.create(
        model="gpt-4o",
        stream=False,  # Prevent streaming for cleaner output
        messages=[
            {
                "role": "system", 
                "content": "You are a helpful assistant that provides concise, direct answers about restaurants and food based on the provided context. Focus on directly answering the user's question."
            },
            {
                "role": "user", 
                "content": f"""Question: {query_sentence}
                
Context:
{context}

Please provide a brief, focused answer based only on the context provided."""
            }
        ]
    )
    print("\nContext:")
    print(context)
    print("\nAnswer:")
    print(response.choices[0].message.content)

# prompt
prompt = "What are the cheapest italian restaurants?"

# Run the RAG function
rag_gpt4o(prompt)