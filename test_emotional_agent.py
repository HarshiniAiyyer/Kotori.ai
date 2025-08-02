import os
import time
from dotenv import load_dotenv
from emotional_agent import emotional_checkin_node

# Load environment variables
load_dotenv()

# Test the emotional agent with a sample input
def test_emotional_agent():
    print("\n===== Testing Emotional Agent =====\n")
    
    # Sample user inputs to test
    test_inputs = [
        "I feel sad today. My children have all moved out and the house feels empty.",
        "I don't know what to do with myself now that the kids are gone.",
        "I miss hearing my children's voices in the house."
    ]
    
    for i, test_input in enumerate(test_inputs):
        print(f"\n\nTEST CASE #{i+1}\n{'-'*50}")
        print(f"User input: {test_input}\n")
        
        # Create state dictionary
        state = {"input": test_input}
        
        # Call the emotional agent
        try:
            print("Generating response...")
            result_state = emotional_checkin_node(state)
            print("\nEmotional Agent Response:\n")
            print(result_state["response"])
            print("\n" + "-"*50)
            
            # Pause between tests
            if i < len(test_inputs) - 1:
                time.sleep(2)
                
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    test_emotional_agent()