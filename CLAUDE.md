# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a photo indexing and search system built with Python that uses CLIP embeddings for semantic image search. The system generates embeddings for images, builds a FAISS index for efficient similarity search, and provides a Streamlit web interface for querying images using natural language.

## Core Architecture

The system consists of three main components:

1. **Embedding Generation** (`generate.py`): Processes images recursively from a directory, extracts CLIP embeddings using OpenAI's CLIP model, and saves them to JSON with metadata
2. **Index Building** (`index.py`): Loads embeddings from JSON and builds a FAISS index using AutoFAISS for efficient similarity search
3. **Query Interface** (`query.py`): Streamlit web app that loads the model and index, processes text queries, and displays matching images

## Key Dependencies

- `open_clip` - For CLIP model and image/text embeddings
- `faiss-cpu` - For efficient similarity search indexing
- `autofaiss` - For automatic FAISS index optimization
- `streamlit` - For web interface
- `torch` - For deep learning operations
- `pillow_heif` - For HEIC image format support

## Common Development Tasks

### Running the Complete Pipeline

#### Option 1: FAISS (Original)
1. **Generate embeddings for images**:
   ```bash
   python generate.py /path/to/images
   ```
   This creates `image_embeddings.json` with embeddings and metadata for all images in the directory.

2. **Build the search index**:
   ```bash
   python index.py
   ```
   This creates `knn.index` and `index_infos.json` files for efficient search.

3. **Launch the search interface**:
   ```bash
   streamlit run query.py
   ```
   This starts the web interface for searching images with natural language queries.

#### Option 2: AgentDB (New)
1. **Set up AgentDB configuration**:
   ```bash
   cp config.env.example .env
   # Edit .env with your AgentDB credentials
   ```

2. **Set up and test AgentDB connection**:
   ```bash
   python setup_agentdb.py
   ```

3. **Generate embeddings for images**:
   ```bash
   python generate.py /path/to/images
   ```

4. **Build the AgentDB index**:
   ```bash
   python index_agentdb.py
   ```
   This stores embeddings in AgentDB instead of local FAISS files.

5. **Launch the AgentDB search interface**:
   ```bash
   streamlit run query_agentdb.py
   ```

### Device Acceleration

The system automatically detects and uses available accelerators:
- CUDA (if available)
- MPS (Apple Silicon)
- Falls back to CPU

### File Structure

#### Core Files
- `generate.py` - Image processing and embedding generation
- `image_embeddings.json` - Generated embeddings with metadata

#### FAISS Implementation (Original)
- `index.py` - FAISS index building
- `query.py` - Streamlit search interface with FAISS
- `knn.index` - FAISS similarity search index
- `index_infos.json` - Index configuration and metadata

#### AgentDB Implementation (New)
- `agentdb.py` - Python SDK for AgentDB
- `index_agentdb.py` - AgentDB index building
- `query_agentdb.py` - Streamlit search interface with AgentDB
- `setup_agentdb.py` - Setup and configuration helper
- `config.env.example` - Environment configuration template

## Technical Details

### Model Configuration
- Uses `hf-hub:laion/CLIP-ViT-g-14-laion2B-s12B-b42K` CLIP model
- Supports common image formats: PNG, JPG, JPEG, GIF, BMP, HEIC
- Extracts EXIF metadata during processing

### Index Configuration

#### FAISS Implementation
- Uses AutoFAISS for automatic optimization
- Index is memory-mapped for efficient loading
- Supports up to 10 search results per query

#### AgentDB Implementation
- Stores embeddings in SQLite database via AgentDB
- Uses cosine similarity for search ranking
- Supports configurable result limits
- Requires AgentDB API key and token for authentication

### Error Handling
- Graceful handling of corrupted images during generation
- Automatic fallback for missing EXIF data
- Proper JSON serialization for various data types
- AgentDB error handling with detailed error messages and response logging

## Environment Variables (AgentDB)

When using the AgentDB implementation, set these environment variables:

```bash
AGENTDB_BASE_URL=https://api.agentdb.dev
AGENTDB_API_KEY=your_api_key_here
AGENTDB_TOKEN=your_token_here
AGENTDB_DB_NAME=photoindex
```

## Key Differences: FAISS vs AgentDB

| Feature | FAISS | AgentDB |
|---------|-------|---------|
| Storage | Local files | Cloud database |
| Search | Optimized vector search | Cosine similarity in Python |
| Setup | No configuration | Requires API credentials |
| Scalability | Memory limited | Cloud scalable |
| Persistence | Local filesystem | Cloud persistence |