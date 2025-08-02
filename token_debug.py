import os
from dotenv import load_dotenv
import requests

# Load environment
load_dotenv()
hf_token = os.getenv("HUGGINGFACE_API_TOKEN")

print("🔍 DETAILED TOKEN DEBUG")
print("=" * 40)
print(f"Token exists: {bool(hf_token)}")
print(f"Token length: {len(hf_token) if hf_token else 0}")
print(f"Token starts with 'hf_': {hf_token.startswith('hf_') if hf_token else False}")
print(f"Token preview: {hf_token[:15]}...{hf_token[-10:] if len(hf_token) > 25 else hf_token}")

# Test different API endpoints
endpoints_to_test = [
    "https://huggingface.co/api/whoami",
    "https://api-inference.huggingface.co/models/gpt2",  # Different endpoint
]

for endpoint in endpoints_to_test:
    print(f"\n🌐 Testing: {endpoint}")
    try:
        headers = {"Authorization": f"Bearer {hf_token}"}
        response = requests.get(endpoint, headers=headers, timeout=10)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text[:200]}")
    except Exception as e:
        print(f"Error: {e}")

# Test manual curl equivalent
print(f"\n🛠️ Manual test command:")
print(f"curl -H 'Authorization: Bearer {hf_token}' https://huggingface.co/api/whoami")