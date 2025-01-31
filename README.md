# Runtime AI Response Generator

This project implements a Monte Carlo Tree Search (MCTS) algorithm to generate AI responses using OpenAI's GPT models. The algorithm explores different possible responses and selects the most promising path based on simulated playouts.

## Setup

1. Create and activate a virtual environment:

```bash
python -m venv env
source env/bin/activate # On Windows, use: env\Scripts\activate
```

2. Install required packages:
```bash
pip install openai python-dotenv numpy
```

3. Create a `.env` file in the project root with your OpenAI API key:

```
OPENAI_API_KEY=your_openai_api_key_here
```

4. Run the example:
bash
python example_usage.py

## Project Structure

### MCTS Implementation (mcts_llm.py)

The `LLMMCTS` class implements the Monte Carlo Tree Search algorithm with the following key components:

#### Core MCTS Methods

- `search(conversation_history, num_simulations)`: Main entry point that performs MCTS for the specified number of simulations. Returns the best path found.

- `select(node)`: Selection phase - traverses the tree from root to leaf using UCB1 formula to balance exploration and exploitation.

- `expand(node, conversation_history)`: Expansion phase - adds new child nodes to the selected leaf node using the language model's predictions.

- `simulate(node, conversation_history, depth)`: Simulation phase - performs a random playout from the node to estimate its value.

- `backpropagate(node, score)`: Backpropagation phase - updates node statistics (visits and values) along the path from the node to root.

#### Token Generation and Processing

- `get_next_tokens(conversation_history, max_tokens)`: Queries the OpenAI API to get possible next tokens and their probabilities. Makes two types of API calls:
  - Short, focused completion (temperature=0.7)
  - Longer, more diverse completion (temperature=1.2)

#### Tree Visualization

- `visualize_tree(max_depth, min_visits)`: Creates a string visualization of the search tree showing:
  - Node content (tokens)
  - Visit counts
  - Value scores
  - Probabilities
  - Success rates

#### Helper Methods

- `_get_conversation_text(node)`: Reconstructs the conversation text from root to the given node.
- `_get_best_path(node)`: Gets the path from root to the given node.
- `_visualize_node(node, depth, max_depth, min_visits)`: Helper for tree visualization.

### Node Class

The `Node` class represents a state in the search tree with:
- Word/token content
- Parent and children references
- Visit count and value statistics
- Log probability of the token
- UCB1 calculation for selection

## How It Works

1. The algorithm starts with an empty root node
2. For each simulation:
   - **Selection**: Traverse tree using UCB1 formula
   - **Expansion**: Add new possible tokens as children
   - **Simulation**: Play out a random sequence to estimate value
   - **Backpropagation**: Update node statistics

3. The final path is chosen by selecting nodes with highest visit counts

## Notes

- The algorithm makes multiple API calls during the search process, which can consume API credits quickly
- Adjust `num_simulations`, `depth`, and other parameters based on your needs
- Higher temperatures and penalties can be used to encourage more diverse responses
