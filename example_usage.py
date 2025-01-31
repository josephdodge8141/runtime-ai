import openai
import os
from dotenv import load_dotenv
from mcts_llm import LLMMCTS

# Load environment variables from .env file
load_dotenv()

# Initialize the OpenAI client with API key from .env
client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Create the MCTS instance with higher temperature
mcts = LLMMCTS(client, "You are a helpful assistant.", temperature=1.0)

# Define the conversation history
conversation_history = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"}
]

# Perform the search with more simulations for better exploration
best_path = mcts.search(conversation_history, num_simulations=12)

# Print the path
print("\nBest response path:")
print("-" * 40)
print(" -> ".join(best_path))
print("-" * 40)
print("Complete response:", " ".join(best_path))

# Print the tree visualization with all visits
print("\nTree Analysis:")
print(mcts.visualize_tree(max_depth=4, min_visits=1))

# Print some statistics
total_nodes = sum(1 for node in mcts.root.children.values())
print(f"\nStatistics:")
print(f"Total top-level choices: {total_nodes}")
print(f"Total root visits: {mcts.root.visits}")
print(f"Average visits per top-level choice: {mcts.root.visits / total_nodes if total_nodes else 0:.2f}") 