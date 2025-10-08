import requests
import json

def test_chatbot():
    """Interactive testing of the chatbot."""
    BASE_URL = "http://localhost:8000"
    
    test_queries = [
        # Basic Program Understanding
        "What is the basic earning rate for points?",
        "How can members track their points?",
        "What are the minimum redemption points?",
        
        # Assignment 1 Related
        "What are common reasons for customer churn?",
        "What metrics are used to measure program success?",
        "How effective are early reward opportunities?",
        
        # Assignment 2 Related
        "What special rewards do VIP members receive?",
        "How do surprise rewards affect customer engagement?",
        "What are examples of surprise and delight features?",
        
        # Edge Cases
        "When do points expire?",
        "Can points be transferred between accounts?",
        "What happens if a reward is not claimed?"
    ]
    
    print("Testing Chatbot Responses...\n")
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        try:
            response = requests.post(
                f"{BASE_URL}/search",
                json={"text": query, "max_results": 3}
            )
            
            if response.status_code == 200:
                results = response.json()
                print("\nResponses:")
                for result in results:
                    print(f"- {result['content']}")
                    print(f"  Source: {result['source']}")
                    print(f"  Relevance: {result['relevance_score']:.2f}\n")
            else:
                print(f"Error: Status code {response.status_code}")
                
        except Exception as e:
            print(f"Error: {str(e)}")
        
        print("-" * 80)
        input("Press Enter to continue...")

if __name__ == "__main__":
    test_chatbot()
