#!/usr/bin/env python3

print("🧪 Testing QnA Agent...")

try:
    from qna_agent import qna_node
    print("✅ QnA agent imported successfully")
    
    state = {
        'input': 'What demographic tends to suffer from Empty Nest Syndrome?', 
        'response': '', 
        'agent': '',
        'intent': ''
    }
    
    print("🔍 Testing with query:", state['input'])
    result = qna_node(state)
    
    print("\n📊 RESULTS:")
    print("Response:", result['response'])
    print("Agent:", result['agent'])
    print("✅ Test completed successfully!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()