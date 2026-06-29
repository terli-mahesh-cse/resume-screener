from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class SemanticMatcher:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        print("Loading model... (first time takes ~30 seconds)")
        self.model = SentenceTransformer(model_name)
        print("Model loaded!")
    
    def match(self, resume_text: str, job_description: str) -> dict:
        # Split into sentences/paragraphs
        resume_chunks = self._chunk_text(resume_text)
        jd_chunks = self._chunk_text(job_description)
        
        # Get embeddings
        resume_emb = self.model.encode(resume_chunks, convert_to_tensor=True)
        jd_emb = self.model.encode(jd_chunks, convert_to_tensor=True)
        
        # Calculate similarity
        similarity_matrix = cosine_similarity(resume_emb.cpu(), jd_emb.cpu())
        max_scores = np.max(similarity_matrix, axis=1)
        
        overall_score = float(np.mean(max_scores)) * 100
        
        # Find matched sections
        matched = []
        for i, chunk in enumerate(resume_chunks):
            best_jd_idx = int(np.argmax(similarity_matrix[i]))
            score = float(similarity_matrix[i][best_jd_idx])
            if score > 0.5:
                matched.append({
                    'resume_text': chunk[:100] + "...",
                    'jd_text': jd_chunks[best_jd_idx][:100] + "...",
                    'score': round(score * 100, 1)
                })
        
        return {
            'overall_match': round(overall_score, 1),
            'matched_sections': matched[:5],
            'total_resume_chunks': len(resume_chunks),
            'total_jd_chunks': len(jd_chunks)
        }
    
    def _chunk_text(self, text: str, max_length: int = 500) -> list:
        sentences = text.replace('\n', '. ').split('. ')
        chunks = []
        current = ""
        for sent in sentences:
            if len(current) + len(sent) < max_length:
                current += sent + ". "
            else:
                if current:
                    chunks.append(current.strip())
                current = sent + ". "
        if current:
            chunks.append(current.strip())
        return chunks if chunks else [text[:500]]