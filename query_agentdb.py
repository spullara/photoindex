import json
import os
import sys
from typing import List, Dict, Tuple, Any
import numpy as np
import open_clip
import streamlit as st
import torch
from PIL import Image
from pillow_heif import register_heif_opener
from agentdb import DatabaseService, AgentDBError
import dotenv
import struct

dotenv.load_dotenv()

# Configuration - you'll need to set these environment variables
AGENTDB_BASE_URL = os.getenv('AGENTDB_BASE_URL', 'https://api.agentdb.dev')
AGENTDB_API_KEY = os.getenv('AGENTDB_API_KEY')
AGENTDB_TOKEN = os.getenv('AGENTDB_TOKEN')
AGENTDB_DB_NAME = os.getenv('AGENTDB_DB_NAME', 'photoindex')

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
if torch.has_mps:
    device = torch.device("mps")

@st.cache_resource
def load_model_and_db():
    """Load the CLIP model and AgentDB connection"""
    register_heif_opener()
    
    # Load CLIP model
    model, _, preprocess = open_clip.create_model_and_transforms(
        'hf-hub:laion/CLIP-ViT-g-14-laion2B-s12B-b42K', 
        device=device
    )
    tokenizer = open_clip.get_tokenizer('hf-hub:laion/CLIP-ViT-g-14-laion2B-s12B-b42K')
    
    # Initialize AgentDB connection
    if not AGENTDB_API_KEY:
        st.error("AGENTDB_API_KEY environment variable is required")
        st.stop()
    if not AGENTDB_TOKEN:
        st.error("AGENTDB_TOKEN environment variable is required")
        st.stop()
    
    db_service = DatabaseService(AGENTDB_BASE_URL, AGENTDB_API_KEY, debug=False)
    db_connection = db_service.connect(AGENTDB_TOKEN, AGENTDB_DB_NAME, 'sqlite')
    
    return model, tokenizer, db_connection

def numpy_to_blob(vector: np.ndarray) -> bytes:
    """Convert numpy array to binary blob for sqlite-vec"""
    # Convert to float32 if not already
    if vector.dtype != np.float32:
        vector = vector.astype(np.float32)

    # Return the raw bytes
    return vector.tobytes()

def blob_to_numpy(blob: bytes) -> np.ndarray:
    """Convert binary blob back to numpy array"""
    # Assuming float32 format - adjust if needed
    return np.frombuffer(blob, dtype=np.float32)

def get_text_embedding(text: str, model, tokenizer) -> np.ndarray:
    """Get embedding for text query as numpy array"""
    with torch.no_grad(), torch.cuda.amp.autocast():
        text_tokens = tokenizer(text).to(device)
        text_embedding = model.encode_text(text_tokens)
        return text_embedding.cpu().detach().numpy()[0]

def search_embeddings(db_connection, query_embedding: np.ndarray, k: int = 10) -> List[Dict[str, Any]]:
    """Search for similar embeddings using sqlite-vec extension"""

    # Convert query embedding to binary format
    query_blob = numpy_to_blob(query_embedding)

    # Use sqlite-vec's cosine distance function for efficient search
    # Note: cosine distance = 1 - cosine similarity, so lower distance = higher similarity
    sql = """
    SELECT
        id,
        path,
        metadata,
        vec_distance_cosine(embedding, ?) as distance
    FROM image_embeddings
    ORDER BY distance ASC
    LIMIT ?
    """

    try:
        result = db_connection.execute({
            'sql': sql,
            'params': [query_blob, k]
        })

        if not result.get('results') or not result['results'][0].get('rows'):
            return []

        # Convert results to the expected format
        similarities = []
        for row in result['results'][0]['rows']:
            row_id, path, metadata_json, distance = row

            # Convert distance back to similarity (similarity = 1 - distance)
            similarity = 1.0 - distance

            similarities.append({
                'id': row_id,
                'path': path,
                'similarity': similarity,
                'metadata': json.loads(metadata_json) if metadata_json else {}
            })

        return similarities

    except Exception as e:
        st.error(f"Vector search error: {e}")
        # Fallback: return empty results
        return []

def search_and_show_images(query: str, model, tokenizer, db_connection, k: int = 10):
    """Search for images and display results"""
    
    if not query.strip():
        st.warning("Please enter a search query")
        return
    
    try:
        # Get query embedding
        query_embedding = get_text_embedding(query, model, tokenizer)
        
        # Search for similar images
        with st.spinner("Searching for similar images..."):
            results = search_embeddings(db_connection, query_embedding, k)
        
        if not results:
            st.info("No results found")
            return
        
        st.success(f"Found {len(results)} similar images")
        
        # Display results
        for i, result in enumerate(results, 1):
            path = result['path']
            similarity = result['similarity']
            metadata = result['metadata']
            
            # Create columns for image and info
            col1, col2 = st.columns([2, 1])
            
            with col1:
                try:
                    if os.path.exists(path):
                        image = Image.open(path)
                        st.image(image, caption=f"#{i} - Similarity: {similarity:.3f}", use_column_width=True)
                    else:
                        st.error(f"Image not found: {path}")
                except Exception as e:
                    st.error(f"Error loading image {path}: {e}")
            
            with col2:
                st.write(f"**Path:** {path}")
                st.write(f"**Similarity:** {similarity:.3f}")
                
                # Show metadata if available
                if metadata:
                    st.write("**Metadata:**")
                    for key, value in metadata.items():
                        if key and value:
                            st.write(f"- {key}: {value}")
                
                st.divider()
    
    except AgentDBError as e:
        st.error(f"Database error: {e}")
    except Exception as e:
        st.error(f"Error during search: {e}")

def show_database_stats(db_connection):
    """Show database statistics"""
    try:
        stats_sql = """
        SELECT 
            COUNT(*) as total_embeddings,
            MIN(created_at) as earliest_created,
            MAX(created_at) as latest_created
        FROM image_embeddings
        """
        
        result = db_connection.execute({'sql': stats_sql})
        
        if result.get('results') and result['results'][0].get('rows'):
            stats = result['results'][0]['rows'][0]
            
            st.sidebar.subheader("Database Statistics")
            st.sidebar.write(f"Total images: {stats[0]}")
            st.sidebar.write(f"Earliest: {stats[1] or 'N/A'}")
            st.sidebar.write(f"Latest: {stats[2] or 'N/A'}")
        
    except Exception as e:
        st.sidebar.error(f"Error fetching stats: {e}")

def main():
    """Main Streamlit app"""
    st.set_page_config(
        page_title="Photo Index Search",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 Photo Index Search")
    st.markdown("Search your photo collection using natural language")
    
    # Load model and database
    try:
        model, tokenizer, db_connection = load_model_and_db()
        
        # Show database stats in sidebar
        show_database_stats(db_connection)
        
        # Main search interface
        st.subheader("Search Photos")
        
        # Search form
        with st.form("search_form"):
            query = st.text_input(
                "Enter your search query",
                placeholder="e.g., sunset over mountains, person with red shirt, cat on a bed"
            )
            
            col1, col2 = st.columns([1, 4])
            with col1:
                num_results = st.slider("Number of results", 1, 20, 10)
            with col2:
                submitted = st.form_submit_button("Search", use_container_width=True)
        
        if submitted and query:
            search_and_show_images(query, model, tokenizer, db_connection, num_results)
        
        # Instructions
        st.markdown("---")
        st.markdown("""
        ### How to use:
        1. Make sure you have run `python index_agentdb.py` to index your images
        2. Set the required environment variables (AGENTDB_API_KEY, AGENTDB_TOKEN)
        3. Enter a natural language description of what you're looking for
        4. Click Search to find similar images
        """)
        
    except Exception as e:
        st.error(f"Error initializing application: {e}")
        st.markdown("""
        ### Setup Instructions:
        1. Set environment variables:
           - `AGENTDB_API_KEY`: Your AgentDB API key
           - `AGENTDB_TOKEN`: Your AgentDB token
           - `AGENTDB_DB_NAME`: Database name (default: photoindex)
        2. Run `python index_agentdb.py` to index your images
        3. Restart this application
        """)

if __name__ == "__main__":
    main()