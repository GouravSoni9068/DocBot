from typing import List, Dict
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class DocumentProcessor:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            min_df=1,
            lowercase=True,
            strip_accents='unicode'
        )
        self.documents = []
        self.vectors = None
        
    def _content_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text snippets."""
        # Convert texts to TF-IDF vectors
        vectors = self.vectorizer.transform([text1, text2])
        # Calculate cosine similarity
        return cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
        
    def _is_redundant(self, text1: str, text2: str) -> bool:
        """Check if two pieces of text contain redundant information."""
        # If either text contains the other, it's redundant
        if text1.lower() in text2.lower() or text2.lower() in text1.lower():
            return True
        # If they're very similar, consider it redundant
        similarity = self._content_similarity(text1, text2)
        return similarity > 0.7  # High similarity threshold
        
    def _is_similar_sentence(self, sentence1: str, sentence2: str) -> bool:
        """Check if two sentences convey similar information."""
        # Normalize the sentences
        s1 = sentence1.lower().strip()
        s2 = sentence2.lower().strip()
        
        # Quick check for exact matches or containment
        if s1 == s2 or s1 in s2 or s2 in s1:
            return True
            
        # Calculate word overlap
        words1 = set(s1.split())
        words2 = set(s2.split())
        overlap = len(words1.intersection(words2)) / min(len(words1), len(words2))
        
        # If there's significant word overlap, check semantic similarity
        if overlap > 0.5:
            similarity = self._content_similarity(s1, s2)
            return similarity > 0.6
            
        return False

    def load_documents(self, json_file: str) -> List[Dict]:
        """Load documents from JSON file."""
        with open(json_file, 'r', encoding='utf-8') as f:
            self.documents = json.load(f)
        return self.documents

    def process_documents(self, documents: List[Dict]) -> None:
        """Create TF-IDF vectors for documents."""
        if not documents:
            raise ValueError("No documents provided")
            
        self.documents = documents
        texts = []
        for doc in documents:
            # Extract text from title and content
            title = doc.get('title', '').strip()
            content = doc.get('content', '').strip()
            
            # Combine texts with space between
            combined_text = f"{title} {content}".strip()
            if combined_text:  # Only add if we have text
                texts.append(combined_text)
        
        if not texts:
            raise ValueError("No valid text found in documents")
        
        # Configure vectorizer to accept custom stop words and handle empty docs
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            min_df=1,
            max_df=1.0,
            strip_accents='unicode',
            lowercase=True
        )
        
        try:
            self.vectors = self.vectorizer.fit_transform(texts)
            print(f"Successfully processed {len(texts)} documents")
        except Exception as e:
            print(f"Error processing documents: {str(e)}")
            raise

    def search_documents(self, query: str, k: int = 3) -> List[Dict]:
        """Search for relevant documents based on query."""
        if self.vectors is None:
            raise ValueError("Documents not processed. Run process_documents first.")

        # Encode query
        query_vector = self.vectorizer.transform([query])

        # Calculate similarities for both whole documents and individual sentences
        doc_similarities = cosine_similarity(query_vector, self.vectors)[0]
        
        # Process documents to find most relevant content
        results = []
        seen_sentences = set()
        
        # First, get the most relevant documents
        doc_indices = np.argsort(doc_similarities)[::-1]
        for doc_idx in doc_indices:
            if doc_similarities[doc_idx] < 0.15:  # Document must be somewhat relevant
                continue
                
            doc = self.documents[doc_idx]
            content = doc['content']
            
            # Split into sentences and find most relevant ones
            sentences = [s.strip() for s in content.split('.') if s.strip()]
            if not sentences:
                continue
                
            sentence_vectors = self.vectorizer.transform(sentences)
            sentence_similarities = cosine_similarity(query_vector, sentence_vectors)[0]
            
            # Get relevant sentences that haven't been used yet
            relevant_sentences = []
            for sent_idx in np.argsort(sentence_similarities)[::-1]:
                if sentence_similarities[sent_idx] < 0.2:  # Sentence must be relevant
                    continue
                    
                sentence = sentences[sent_idx]
                # Check if this sentence is too similar to ones we've already used
                if not any(self._is_similar_sentence(sentence, seen) for seen in seen_sentences):
                    relevant_sentences.append(sentence)
                    seen_sentences.add(sentence)
                    if len(relevant_sentences) >= 2:  # Max 2 sentences per document
                        break
            
            if relevant_sentences:
                results.append({
                    'content': '. '.join(relevant_sentences) + '.',
                    'metadata': {
                        'source': doc['url'],
                        'title': doc['title']
                    },
                    'relevance_score': max(sentence_similarities)  # Use highest sentence similarity
                })
                
            if len(results) >= k:
                break
                
        return results
        
        results = []
        for idx in top_k_indices:
            doc = self.documents[idx]
            score = float(similarities[idx])
            
            # Only include if score is good enough
            if score > threshold:
                # Extract relevant section from content
                content = doc['content']
                
                # Find most relevant part of the content
                sentences = [s.strip() for s in content.split('.') if s.strip()]
                sentence_vectors = self.vectorizer.transform(sentences)
                sentence_scores = cosine_similarity(query_vector, sentence_vectors)[0]
                
                # Get most relevant sentence and its context
                best_idx = np.argmax(sentence_scores)
                best_score = sentence_scores[best_idx]
                
                # Only include sentences that are highly relevant to the query
                relevant_sentences = []
                if best_score > 0.3:  # Main sentence must be highly relevant
                    relevant_sentences.append(sentences[best_idx])
                    
                    # Add context only if it's closely related to the query
                    if len(sentences) > 1:
                        context_scores = [(i, score) for i, score in enumerate(sentence_scores) 
                                       if i != best_idx and score > 0.4 * best_score]  # Must be at least 40% as relevant as best
                        if context_scores:
                            # Sort by relevance and position relative to best sentence
                            context_scores.sort(key=lambda x: (-x[1], abs(x[0] - best_idx)))
                            # Take only the most relevant context
                            context_idx = context_scores[0][0]
                            # Only add if it adds new information
                            if not self._is_redundant(sentences[context_idx], sentences[best_idx]):
                                relevant_sentences.append(sentences[context_idx])
                
                relevant_content = '. '.join(relevant_sentences) + '.'
                
                # Skip this result if we didn't find any highly relevant content
                if not relevant_sentences:
                    continue
                
                results.append({
                    'content': relevant_content,
                    'metadata': {
                        'source': doc['url'],
                        'title': doc['title']
                    },
                    'relevance_score': score
                })
        
        return results
