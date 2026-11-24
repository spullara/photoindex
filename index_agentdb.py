import json
import os
import sys
import time
import random
from typing import List, Dict, Any, Optional
from agentdb import DatabaseService, ValidationError, AgentDBError
import dotenv
import numpy as np

dotenv.load_dotenv()

# Configuration - you'll need to set these environment variables
AGENTDB_BASE_URL = os.getenv('AGENTDB_BASE_URL', 'https://api.agentdb.dev')
AGENTDB_API_KEY = os.getenv('AGENTDB_API_KEY')
AGENTDB_TOKEN = os.getenv('AGENTDB_TOKEN')
AGENTDB_DB_NAME = os.getenv('AGENTDB_DB_NAME', 'photoindex')

# Progress tracking configuration
PROGRESS_FILE_PREFIX = "agentdb_upload_progress"
MAX_RETRIES = 3
INITIAL_RETRY_DELAY = 1.0  # seconds
MAX_RETRY_DELAY = 60.0  # seconds

def numpy_to_blob(vector: np.ndarray) -> bytes:
    """Convert numpy array to binary blob for sqlite-vec"""
    # Convert to float32 if not already
    if vector.dtype != np.float32:
        vector = vector.astype(np.float32)

    # Return the raw bytes
    return vector.tobytes()

def get_progress_filename(embeddings_file: str) -> str:
    """Generate progress file name based on embeddings file"""
    base_name = os.path.splitext(os.path.basename(embeddings_file))[0]
    return f"{PROGRESS_FILE_PREFIX}_{base_name}.json"

def load_progress(progress_file: str) -> Dict[str, Any]:
    """Load progress from file"""
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'r') as f:
                progress = json.load(f)
                print(f"Loaded progress: {progress['completed_batches']} batches completed out of {progress['total_batches']}")
                return progress
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Could not load progress file {progress_file}: {e}")
            print("Starting from beginning...")
    return {"completed_batches": 0, "total_batches": 0, "successful_uploads": 0, "initial_count": 0}

def save_progress(progress_file: str, completed_batches: int, total_batches: int, successful_uploads: int, initial_count: int = 0) -> None:
    """Save current progress to file"""
    progress = {
        "completed_batches": completed_batches,
        "total_batches": total_batches,
        "successful_uploads": successful_uploads,
        "initial_count": initial_count,
        "last_updated": time.time()
    }
    try:
        with open(progress_file, 'w') as f:
            json.dump(progress, f, indent=2)
    except Exception as e:
        print(f"Warning: Could not save progress: {e}")

def retry_with_backoff(max_retries: int = MAX_RETRIES, initial_delay: float = INITIAL_RETRY_DELAY):
    """Decorator to add exponential backoff retry logic"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except (AgentDBError, ValidationError) as e:
                    last_exception = e
                    if attempt == max_retries:
                        print(f"Failed after {max_retries + 1} attempts: {e}")
                        raise e

                    # Calculate delay with exponential backoff and jitter
                    delay = min(initial_delay * (2 ** attempt), MAX_RETRY_DELAY)
                    jitter = random.uniform(0.1, 0.3) * delay
                    total_delay = delay + jitter

                    print(f"Attempt {attempt + 1} failed: {e}. Retrying in {total_delay:.1f} seconds...")
                    time.sleep(total_delay)
                except Exception as e:
                    # For non-AgentDB errors, don't retry
                    print(f"Non-retryable error: {e}")
                    raise e

            # This should never be reached, but just in case
            raise last_exception
        return wrapper
    return decorator



def migrate_embeddings_to_blob(db_connection) -> None:
    """Migrate existing TEXT embeddings to BLOB format"""
    try:
        # Check if we have any TEXT embeddings to migrate
        check_sql = """
        SELECT id, path, embedding
        FROM image_embeddings
        WHERE typeof(embedding) = 'text'
        LIMIT 1
        """
        result = db_connection.execute({'sql': check_sql})

        if not result.get('results') or not result['results'][0].get('rows'):
            print("No TEXT embeddings found to migrate")
            return

        print("Found TEXT embeddings, migrating to BLOB format...")

        # Get all TEXT embeddings
        get_all_sql = """
        SELECT id, path, embedding, metadata
        FROM image_embeddings
        WHERE typeof(embedding) = 'text'
        """
        result = db_connection.execute({'sql': get_all_sql})

        if not result.get('results') or not result['results'][0].get('rows'):
            return

        # Convert and update each embedding
        update_statements = []
        for row in result['results'][0]['rows']:
            row_id, path, embedding_json, metadata = row
            try:
                # Parse JSON embedding and convert to BLOB
                embedding_list = json.loads(embedding_json)
                embedding_array = np.array(embedding_list, dtype=np.float32)
                embedding_blob = numpy_to_blob(embedding_array)

                update_statements.append({
                    'sql': 'UPDATE image_embeddings SET embedding = ? WHERE id = ?',
                    'params': [embedding_blob, row_id]
                })

            except (json.JSONDecodeError, ValueError) as e:
                print(f"Error migrating embedding for {path}: {e}")
                continue

        if update_statements:
            print(f"Migrating {len(update_statements)} embeddings to BLOB format...")
            db_connection.execute(update_statements)
            print("Migration completed successfully")

    except Exception as e:
        print(f"Migration error: {e}")

def create_embeddings_table(db_connection) -> None:
    """Create the embeddings table if it doesn't exist"""
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS image_embeddings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        path TEXT NOT NULL UNIQUE,
        embedding BLOB NOT NULL,
        metadata TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    );
    """
    
    # Create index for faster path lookups
    create_index_sql = """
    CREATE INDEX IF NOT EXISTS idx_path ON image_embeddings(path);
    """
    db_service = DatabaseService(AGENTDB_BASE_URL, AGENTDB_API_KEY, debug=True)
    db_connection2 = db_service.connect(AGENTDB_TOKEN, AGENTDB_DB_NAME, 'sqlite')

    db_connection2.execute([
        {'sql': create_table_sql},
        {'sql': create_index_sql}
    ])

    print("Created embeddings table and index")
    
    # Show the tables that exist in the database now
    list_tables_sql = "SELECT name FROM sqlite_master WHERE type='table';"
    result = db_connection2.execute({'sql': list_tables_sql})
    print("Tables in database:", result['results'][0]['rows'])

def insert_embeddings(db_connection, embeddings_data: List[Dict[str, Any]], progress_file: Optional[str] = None) -> None:
    """Insert embeddings into the database with progress tracking and retry logic"""
    statements = []

    print("Creating insert statements")
    for item in embeddings_data:
        # Convert embedding numpy array to binary blob for sqlite-vec
        embedding_array = np.array(item['embedding'], dtype=np.float32)
        embedding_blob = numpy_to_blob(embedding_array)
        metadata_json = json.dumps(item.get('metadata', {}))

        # Use INSERT OR REPLACE to handle duplicates
        sql = """
        INSERT OR REPLACE INTO image_embeddings (path, embedding, metadata)
        VALUES (?, ?, ?)
        """
        statements.append({
            'sql': sql,
            'params': [item['path'], embedding_blob, metadata_json]
        })

    # Load existing progress
    progress = {"completed_batches": 0, "total_batches": 0, "successful_uploads": 0}
    if progress_file:
        progress = load_progress(progress_file)

    # Execute in batches to avoid overwhelming the API
    batch_size = 50
    total_batches = (len(statements) + batch_size - 1) // batch_size
    progress["total_batches"] = total_batches

    print(f"Inserting {len(statements)} embeddings in batches of {batch_size}")
    if progress["completed_batches"] > 0:
        print(f"Resuming from batch {progress['completed_batches'] + 1}")

    @retry_with_backoff()
    def execute_batch_with_transaction(batch_statements: List[Dict], batch_num: int) -> int:
        """Execute a batch of statements within a transaction"""
        # Wrap batch in transaction
        transaction_statements = [
            {'sql': 'BEGIN TRANSACTION'},
            *batch_statements,
            {'sql': 'COMMIT'}
        ]

        try:
            result = db_connection.execute(transaction_statements)

            # Debug: print result structure for troubleshooting
            if batch_num <= 5:  # Only print for a few batches to avoid spam
                print(f"Debug batch {batch_num}: result keys = {result.keys() if result else 'None'}")
                if result and 'results' in result:
                    print(f"Debug batch {batch_num}: results length = {len(result['results'])}")

            # Count successful inserts more carefully
            if not result or 'results' not in result:
                return len(batch_statements)  # Assume success if no error info

            results = result['results']
            if len(results) < len(transaction_statements):
                return len(batch_statements)  # Assume success if results are truncated

            # Count successful INSERT statements (skip BEGIN at index 0 and COMMIT at end)
            successful_inserts = 0
            for i in range(1, len(results) - 1):  # Skip first (BEGIN) and last (COMMIT)
                if i - 1 < len(batch_statements):  # Make sure we're within batch bounds
                    result_item = results[i]
                    if not result_item.get('error'):
                        successful_inserts += 1
                    else:
                        print(f"Error in statement {i - 1}: {result_item['error']}")

            return successful_inserts

        except Exception as e:
            # Try to rollback on error
            try:
                db_connection.execute([{'sql': 'ROLLBACK'}])
            except:
                pass  # Rollback might fail if transaction was already rolled back
            raise e

    # Get initial count before processing (or use stored value if resuming)
    if progress["completed_batches"] > 0 and "initial_count" in progress:
        initial_count = progress["initial_count"]
        print(f"Resuming: Using stored initial count: {initial_count}")
    else:
        try:
            count_result = db_connection.execute([{'sql': 'SELECT COUNT(*) as count FROM image_embeddings'}])
            initial_count = count_result['results'][0]['rows'][0]['count'] if count_result.get('results') else 0
            print(f"Initial embeddings count: {initial_count}")
        except Exception as e:
            print(f"✗ ERROR: Could not get initial database count: {e}")
            print("Cannot proceed without knowing the starting state.")
            sys.exit(1)

    # Process batches, starting from where we left off
    for i in range(progress["completed_batches"] * batch_size, len(statements), batch_size):
        batch_num = i // batch_size + 1
        batch = statements[i:i + batch_size]

        try:
            successful_in_batch = execute_batch_with_transaction(batch, batch_num)
            progress["completed_batches"] = batch_num
            progress["successful_uploads"] += successful_in_batch

            # Verify actual count in database after batch
            try:
                count_result = db_connection.execute([{'sql': 'SELECT COUNT(*) as count FROM image_embeddings'}])
                actual_count = count_result['results'][0]['rows'][0]['count'] if count_result.get('results') else 0
                expected_count = initial_count + progress["successful_uploads"]

                if actual_count == expected_count:
                    print(f"✓ Completed batch {batch_num}/{total_batches} ({successful_in_batch} records) - DB count: {actual_count} ✓")
                else:
                    print(f"✗ ERROR: Database count mismatch after batch {batch_num}")
                    print(f"   Expected: {expected_count}, Actual: {actual_count}")
                    print(f"   Difference: {actual_count - expected_count}")
                    print(f"   Progress saved in: {progress_file}")
                    sys.exit(1)

            except Exception as count_error:
                print(f"✗ ERROR: Could not verify database count after batch {batch_num}: {count_error}")
                print(f"   Progress saved in: {progress_file}")
                sys.exit(1)

            # Save progress after each successful batch
            if progress_file:
                save_progress(progress_file, progress["completed_batches"],
                            progress["total_batches"], progress["successful_uploads"], initial_count)

        except Exception as e:
            print(f"✗ Failed to process batch {batch_num}: {e}")
            raise e

    # Final verification
    try:
        final_count_result = db_connection.execute([{'sql': 'SELECT COUNT(*) as count FROM image_embeddings'}])
        final_count = final_count_result['results'][0]['rows'][0]['count'] if final_count_result.get('results') else 0
        expected_final_count = initial_count + progress['successful_uploads']

        print(f"Upload process completed!")
        print(f"Final database count: {final_count} embeddings")
        print(f"Processed {progress['completed_batches']} batches")
        print(f"Expected final count: {expected_final_count}")

        if final_count != expected_final_count:
            print(f"✗ ERROR: Final count verification failed!")
            print(f"   Expected: {expected_final_count}")
            print(f"   Actual: {final_count}")
            print(f"   Difference: {final_count - expected_final_count}")
            sys.exit(1)
        else:
            print("✓ Final database count verification passed!")

    except Exception as e:
        print(f"✗ ERROR: Could not verify final database count: {e}")
        print("Upload process completed but final verification failed.")
        sys.exit(1)

def create_similarity_search_functions(db_connection) -> None:
    """Create helper functions for similarity search"""
    
    # Note: SQLite doesn't have native vector similarity functions
    # We'll need to implement cosine similarity in the query layer
    # For now, we'll create a simple lookup table structure
    
    print("Database schema created successfully")

def build_agentdb_index(embeddings_file: str = "image_embeddings.json") -> None:
    """Build AgentDB index from embeddings JSON file with progress tracking"""

    # Validate environment variables
    if not AGENTDB_API_KEY:
        raise ValueError("AGENTDB_API_KEY environment variable is required")
    if not AGENTDB_TOKEN:
        raise ValueError("AGENTDB_TOKEN environment variable is required")

    # Setup progress tracking
    progress_file = get_progress_filename(embeddings_file)

    # Load embeddings data
    print(f"Loading embeddings from {embeddings_file}")
    with open(embeddings_file, "r") as f:
        embeddings_data = json.load(f)

    print(f"Loaded {len(embeddings_data)} embeddings")

    # Initialize AgentDB service
    db_service = DatabaseService(AGENTDB_BASE_URL, AGENTDB_API_KEY, debug=False)

    try:
        # Connect to database (will create if doesn't exist)
        db_connection = db_service.connect(AGENTDB_TOKEN, AGENTDB_DB_NAME, 'sqlite')

        # Create table schema
        create_embeddings_table(db_connection)

        # Migrate any existing TEXT embeddings to BLOB format
        migrate_embeddings_to_blob(db_connection)

        # Insert embeddings with progress tracking
        insert_embeddings(db_connection, embeddings_data, progress_file)

        # Create helper functions
        create_similarity_search_functions(db_connection)

        print(f"Successfully indexed {len(embeddings_data)} embeddings in AgentDB")
        print(f"Database: {AGENTDB_DB_NAME}")
        print(f"Table: image_embeddings")

        # Clean up progress file on successful completion
        if os.path.exists(progress_file):
            os.remove(progress_file)
            print(f"Cleaned up progress file: {progress_file}")

    except AgentDBError as e:
        print(f"AgentDB Error: {e}")
        if e.response:
            print(f"Response details: {e.response}")
        print(f"Progress saved in: {progress_file}")
        print("You can restart the script to resume from where it left off.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        print(f"Progress saved in: {progress_file}")
        print("You can restart the script to resume from where it left off.")
        sys.exit(1)

def get_embedding_stats(db_connection) -> Dict[str, Any]:
    """Get statistics about the embeddings in the database"""
    stats_sql = """
    SELECT 
        COUNT(*) as total_embeddings,
        MIN(created_at) as earliest_created,
        MAX(created_at) as latest_created
    FROM image_embeddings
    """
    
    result = db_connection.execute({'sql': stats_sql})
    return result['results'][0]['rows'][0] if result['results'] else {}

def cleanup_old_progress_files() -> None:
    """Clean up old progress files"""
    try:
        for filename in os.listdir('.'):
            if filename.startswith(PROGRESS_FILE_PREFIX) and filename.endswith('.json'):
                file_age = time.time() - os.path.getmtime(filename)
                # Remove progress files older than 7 days
                if file_age > 7 * 24 * 3600:
                    os.remove(filename)
                    print(f"Removed old progress file: {filename}")
    except Exception as e:
        print(f"Warning: Could not clean up old progress files: {e}")

def main():
    """Main function to build the index"""
    embeddings_file = "image_embeddings.json"

    # Check if input file exists
    if not os.path.exists(embeddings_file):
        print(f"Error: {embeddings_file} not found")
        print("Please run generate.py first to create the embeddings file")
        sys.exit(1)

    # Clean up old progress files
    cleanup_old_progress_files()

    # Check for existing progress
    progress_file = get_progress_filename(embeddings_file)
    if os.path.exists(progress_file):
        progress = load_progress(progress_file)
        if progress["completed_batches"] > 0:
            print(f"Found existing progress file. Will resume from batch {progress['completed_batches'] + 1}")

    print("Building AgentDB index...")
    build_agentdb_index(embeddings_file)

    # Show stats
    db_service = DatabaseService(AGENTDB_BASE_URL, AGENTDB_API_KEY, debug=False)
    db_connection = db_service.connect(AGENTDB_TOKEN, AGENTDB_DB_NAME, 'sqlite')
    stats = get_embedding_stats(db_connection)

    print("\nIndex Statistics:")
    print(f"Total embeddings: {stats.get('total_embeddings', 'Unknown')}")
    print(f"Earliest created: {stats.get('earliest_created', 'Unknown')}")
    print(f"Latest created: {stats.get('latest_created', 'Unknown')}")

if __name__ == "__main__":
    main()