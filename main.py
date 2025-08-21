from robyn import Robyn
from robyn.responses import Response
from embedding import embedding_service
import json

app = Robyn(__file__)

@app.get("/")
async def h(request):
    return "Hello, world!"

from robyn import Robyn
from embedding import embedding_service
import json

app = Robyn(__file__)

@app.get("/")
async def h(request):
    return "Hello, world!"

@app.post("/embeddings")
async def get_embeddings(request):
    """Endpoint to generate embeddings for text"""
    # Parse JSON body
    try:
        body = request.body
        if isinstance(body, bytes):
            body = body.decode('utf-8')
        
        if not body.strip():
            return {"error": "Empty request body"}
        
        data = json.loads(body)
        
    except json.JSONDecodeError as e:
        return {"error": f"Invalid JSON: {str(e)}"}
    except Exception as e:
        return {"error": f"Request parsing error: {str(e)}"}
    
    # Validate and process request
    try:
        # Check if it's a single text or batch
        if "text" in data:
            # Single text embedding
            text = data["text"]
            if not isinstance(text, str):
                return {"error": "Text must be a string"}
            
            if not text.strip():
                return {"error": "Text cannot be empty"}
            
            embeddings = embedding_service.get_embeddings(text)
            return {
                "text": text,
                "embeddings": embeddings,
                "dimension": len(embeddings)
            }
        
        elif "texts" in data:
            # Batch text embeddings
            texts = data["texts"]
            if not isinstance(texts, list):
                return {"error": "Texts must be a list"}
            
            if not texts:
                return {"error": "Texts list cannot be empty"}
            
            if not all(isinstance(t, str) for t in texts):
                return {"error": "All items in texts must be strings"}
            
            embeddings = embedding_service.get_batch_embeddings(texts)
            return {
                "texts": texts,
                "embeddings": embeddings,
                "count": len(embeddings),
                "dimension": len(embeddings[0]) if embeddings else 0
            }
        
        else:
            return {"error": "Request must contain either 'text' or 'texts' field"}
    
    except Exception as e:
        return {"error": f"Embedding generation error: {str(e)}"}

@app.get("/health")
async def health_check(request):
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model": "all-MiniLM-L6-v2",
        "embedding_service_initialized": embedding_service._initialized
    }

app.start(port=8080)