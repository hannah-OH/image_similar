from numpy import dot
from numpy.linalg import norm

class Cosine_similarity:
    def _cos_sin(A, B):
        return dot(A, B) / (norm(A) * norm(B))