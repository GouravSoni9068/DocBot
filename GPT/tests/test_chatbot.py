import requests
import json
from typing import List, Dict
import unittest

class TestChatbot(unittest.TestCase):
    BASE_URL = "http://localhost:8000"
    
    def send_query(self, query: str) -> List[Dict]:
        """Send a query to the chatbot and return the response."""
        response = requests.post(
            f"{self.BASE_URL}/search",
            json={"text": query, "max_results": 3}
        )
        return response.json()
    
    def assert_response_relevant(self, query: str, expected_keywords: List[str]):
        """Test if response contains relevant information."""
        responses = self.send_query(query)
        
        # Check if we got any responses
        self.assertTrue(len(responses) > 0, f"No response received for query: {query}")
        
        # Check if response contains expected keywords
        response_text = ' '.join(r['content'].lower() for r in responses)
        for keyword in expected_keywords:
            self.assertIn(
                keyword.lower(), 
                response_text, 
                f"Expected keyword '{keyword}' not found in response for query: {query}"
            )
    
    def test_basic_queries(self):
        """Test basic program understanding questions."""
        test_cases = [
            {
                "query": "What is the basic earning rate for points?",
                "keywords": ["1 point", "per $1", "spent"]
            },
            {
                "query": "How can members track their points?",
                "keywords": ["mobile app", "website"]
            },
            {
                "query": "What are the minimum redemption points?",
                "keywords": ["100 points"]
            }
        ]
        
        for case in test_cases:
            self.assert_response_relevant(case["query"], case["keywords"])
    
    def test_assignment1_queries(self):
        """Test queries related to Assignment 1."""
        test_cases = [
            {
                "query": "What are common reasons for customer churn?",
                "keywords": ["redemption thresholds", "delayed reward", "personalized"]
            },
            {
                "query": "What metrics measure program success?",
                "keywords": ["retention rate", "redemption rate", "engagement"]
            }
        ]
        
        for case in test_cases:
            self.assert_response_relevant(case["query"], case["keywords"])
    
    def test_assignment2_queries(self):
        """Test queries related to Assignment 2."""
        test_cases = [
            {
                "query": "What special rewards do VIP members receive?",
                "keywords": ["birthday", "anniversary", "milestone"]
            },
            {
                "query": "How effective are surprise rewards?",
                "keywords": ["25% increase", "engagement"]
            }
        ]
        
        for case in test_cases:
            self.assert_response_relevant(case["query"], case["keywords"])
    
    def test_edge_cases(self):
        """Test edge case queries."""
        test_cases = [
            {
                "query": "When do points expire?",
                "keywords": ["12 months", "inactivity"]
            },
            {
                "query": "Can points be transferred?",
                "keywords": ["transfer", "authorization"]
            }
        ]
        
        for case in test_cases:
            self.assert_response_relevant(case["query"], case["keywords"])

if __name__ == '__main__':
    unittest.main()
