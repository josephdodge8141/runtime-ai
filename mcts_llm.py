import math
import openai
from typing import List, Dict, Optional
import numpy as np

class Node:
    def __init__(self, word: str, parent: Optional['Node'] = None, logprob: float = 0.0):
        self.word = word
        self.parent = parent
        self.children: Dict[str, Node] = {}
        self.visits = 0
        self.value = 0.0
        self.logprob = logprob
        
    def add_child(self, word: str, logprob: float) -> 'Node':
        child = Node(word, self, logprob)
        self.children[word] = child
        return child
    
    def get_ucb(self, exploration_constant: float = 1.414) -> float:
        if self.visits == 0:
            return float('inf')
        
        exploitation = self.value / self.visits
        exploration = exploration_constant * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration

class LLMMCTS:
    def __init__(self, client: openai.OpenAI, system_prompt: str, temperature: float = 0.7):
        self.client = client
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.root = Node("")
        
    def get_next_tokens(self, conversation_history: List[dict], max_tokens: int = 1) -> List[tuple]:
        """Get the next possible tokens and their probabilities"""
        all_tokens = {}
        
        try:
            # Make two different types of calls
            responses = []
            
            # First call: Short, focused completion
            response1 = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=conversation_history,
                max_tokens=1,
                n=5,
                temperature=0.7,
                presence_penalty=0.0,
                frequency_penalty=0.0,
            )
            responses.extend(response1.choices)
            
            # Second call: Slightly longer, more diverse
            response2 = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=conversation_history,
                max_tokens=2,
                n=5,
                temperature=1.2,
                presence_penalty=1.0,
                frequency_penalty=1.0,
            )
            responses.extend(response2.choices)
            
            # Process all responses
            for choice in responses:
                content = choice.message.content.strip()
                if content:
                    # Split into words but keep punctuation
                    tokens = [t for t in content.replace('.', ' .').replace('?', ' ?').split() if t]
                    if tokens:
                        token = tokens[0]
                        if token not in all_tokens:
                            all_tokens[token] = 1
                        else:
                            all_tokens[token] += 1
                
            print(f"\nRaw completions:", [choice.message.content for choice in responses])
            
        except Exception as e:
            print(f"Error in API call: {e}")
            return []
        
        if not all_tokens:
            return []
        
        # Convert counts to probabilities with smoothing
        total_counts = sum(all_tokens.values())
        smoothing_factor = 0.2
        tokens = [(token, (count + smoothing_factor)/(total_counts + smoothing_factor * len(all_tokens))) 
                 for token, count in all_tokens.items()]
        
        # Sort by probability and take top 5
        tokens = sorted(tokens, key=lambda x: x[1], reverse=True)[:5]
        
        # Final normalization
        total_prob = sum(prob for _, prob in tokens)
        tokens = [(token, prob/total_prob) for token, prob in tokens]
        
        print(f"\nFinal tokens and probabilities: {tokens}\n")
        return tokens
    
    def select(self, node: Node) -> Node:
        """Select a promising node to expand"""
        current = node
        while current.children:
            max_ucb = float('-inf')
            best_child = None
            
            for child in current.children.values():
                ucb = child.get_ucb()
                if ucb > max_ucb:
                    max_ucb = ucb
                    best_child = child
            
            current = best_child
        return current
    
    def expand(self, node: Node, conversation_history: List[dict]) -> Node:
        """Expand the selected node with new children"""
        # Get the current conversation context
        current_text = self._get_conversation_text(node)
        current_history = conversation_history + [{"role": "assistant", "content": current_text}]
        
        # Get possible next tokens
        next_tokens = self.get_next_tokens(current_history)
        
        # Add children nodes
        for token, prob in next_tokens:
            if token not in node.children:  # Only add if not already a child
                node.add_child(token, math.log(prob))
        
        if node.children:
            # Return a child weighted by probability
            children = list(node.children.values())
            probs = np.array([math.exp(child.logprob) for child in children])
            probs = probs / np.sum(probs)
            return np.random.choice(children, p=probs)
        
        return node  # Return self if no children were added
    
    def simulate(self, node: Node, conversation_history: List[dict], depth: int = 3) -> float:
        """Simulate a random playout from the node"""
        current = node
        current_text = self._get_conversation_text(current)
        score = 0.0
        
        for d in range(depth):
            current_history = conversation_history + [{"role": "assistant", "content": current_text}]
            next_tokens = self.get_next_tokens(current_history)
            
            if not next_tokens:
                break
                
            # Select next token based on probabilities
            probs = np.array([prob for _, prob in next_tokens])
            probs = probs / np.sum(probs)
            selected_idx = np.random.choice(len(next_tokens), p=probs)
            token, prob = next_tokens[selected_idx]
            
            # Decrease score impact with depth
            score += prob * (0.9 ** d)
            current_text += " " + token
            
            # Stop if we've completed a sentence
            if token in ['.', '!', '?']:
                # Add bonus for completing the sentence
                score += 0.5
                break
            
            # Penalize very long responses
            if len(current_text.split()) >= 8:
                score *= 0.7
                break
        
        # Penalize very short responses
        if len(current_text.split()) < 3:
            score *= 0.5
        
        return score
    
    def backpropagate(self, node: Node, score: float):
        """Update statistics for all nodes in the path"""
        current = node
        while current is not None:
            current.visits += 1
            current.value += score
            current = current.parent
    
    def _get_conversation_text(self, node: Node) -> str:
        """Reconstruct the conversation text from root to node"""
        words = []
        current = node
        while current.parent is not None:
            words.append(current.word)
            current = current.parent
        return " ".join(reversed(words))
    
    def _get_best_path(self, node: Node) -> List[str]:
        """Get the path from root to the given node"""
        path = []
        current = node
        while current.parent is not None:
            path.append(current.word)
            current = current.parent
        return list(reversed(path))

    def search(self, conversation_history: List[dict], num_simulations: int = 100) -> List[str]:
        """Perform MCTS search to find the best response path"""
        for _ in range(num_simulations):
            # Selection
            selected_node = self.select(self.root)
            
            # Expansion
            expanded_node = self.expand(selected_node, conversation_history)
            
            # Simulation
            score = self.simulate(expanded_node, conversation_history)
            
            # Backpropagation
            self.backpropagate(expanded_node, score)
        
        # Find the path with highest visit count
        current = self.root
        path = []
        while current.children:
            current = max(current.children.values(), key=lambda x: x.visits)
            path.append(current.word)
        
        return path

    def _visualize_node(self, node: Node, depth: int = 0, max_depth: int = None, min_visits: int = 1) -> str:
        """Helper method to create a string visualization of the tree from a given node"""
        if max_depth is not None and depth > max_depth:
            return ""
        
        # Skip nodes with too few visits
        if node.visits < min_visits:
            return ""
        
        # Calculate success rate
        success_rate = node.value / node.visits if node.visits > 0 else 0
        
        # Create the current node string with indentation and stats
        result = "  " * depth
        result += f"├─ '{node.word}' "
        result += f"(visits: {node.visits}, "
        result += f"value: {node.value:.3f}, "
        result += f"prob: {math.exp(node.logprob):.3f}, "
        result += f"success: {success_rate:.3f})\n"
        
        # Sort children by visit count
        sorted_children = sorted(
            node.children.values(),
            key=lambda x: (x.visits, x.value),
            reverse=True
        )
        
        # Recursively add children
        for child in sorted_children:
            result += self._visualize_node(child, depth + 1, max_depth, min_visits)
        
        return result

    def visualize_tree(self, max_depth: int = None, min_visits: int = 1) -> str:
        """
        Create a string visualization of the entire tree
        
        Args:
            max_depth: Maximum depth to display (None for unlimited)
            min_visits: Minimum number of visits for a node to be displayed
        """
        result = "MCTS Tree Visualization:\n"
        result += "=" * 50 + "\n"
        result += self._visualize_node(self.root, max_depth=max_depth, min_visits=min_visits)
        result += "=" * 50 + "\n"
        return result 