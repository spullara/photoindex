#!/usr/bin/env python3
"""
Setup script for AgentDB integration
This script helps you set up the AgentDB environment and test the connection
"""

import os
import sys
import json
from agentdb import DatabaseService, ValidationError, AgentDBError

def load_env_file(filename='.env'):
    """Load environment variables from a file"""
    if not os.path.exists(filename):
        return False
    
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                key, value = line.split('=', 1)
                os.environ[key] = value
    return True

def test_agentdb_connection():
    """Test the AgentDB connection with current environment variables"""
    base_url = os.getenv('AGENTDB_BASE_URL', 'https://api.agentdb.dev')
    api_key = os.getenv('AGENTDB_API_KEY')
    token = os.getenv('AGENTDB_TOKEN')
    db_name = os.getenv('AGENTDB_DB_NAME', 'photoindex')
    
    if not api_key:
        print("❌ AGENTDB_API_KEY is not set")
        return False
    
    if not token:
        print("❌ AGENTDB_TOKEN is not set")
        return False
    
    try:
        print(f"🔗 Testing connection to {base_url}")
        print(f"📊 Database: {db_name}")
        
        # Initialize service
        db_service = DatabaseService(base_url, api_key, debug=True)
        
        # Test listing databases
        print("📋 Listing databases...")
        databases = db_service.list_databases(token)
        print(f"✅ Found {len(databases)} databases")
        
        # Test connection
        print(f"🔌 Testing connection to database '{db_name}'...")
        db_connection = db_service.connect(token, db_name, 'sqlite')
        
        # Test a simple query
        result = db_connection.execute({'sql': 'SELECT 1 as test'})
        print("✅ Database connection successful")
        
        return True
        
    except AgentDBError as e:
        print(f"❌ AgentDB Error: {e}")
        if e.response:
            print(f"Response: {json.dumps(e.response, indent=2)}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def setup_database():
    """Set up the database schema"""
    base_url = os.getenv('AGENTDB_BASE_URL', 'https://api.agentdb.dev')
    api_key = os.getenv('AGENTDB_API_KEY')
    token = os.getenv('AGENTDB_TOKEN')
    db_name = os.getenv('AGENTDB_DB_NAME', 'photoindex')
    
    try:
        print("🛠️  Setting up database schema...")
        
        db_service = DatabaseService(base_url, api_key, debug=False)
        db_connection = db_service.connect(token, db_name, 'sqlite')
        
        # Create the embeddings table
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS image_embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT NOT NULL UNIQUE,
            embedding TEXT NOT NULL,
            metadata TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        """
        
        create_index_sql = """
        CREATE INDEX IF NOT EXISTS idx_path ON image_embeddings(path);
        """
        
        db_connection.execute([
            {'sql': create_table_sql},
            {'sql': create_index_sql}
        ])
        
        print("✅ Database schema created successfully")
        return True
        
    except Exception as e:
        print(f"❌ Error setting up database: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 AgentDB Photo Index Setup")
    print("=" * 40)
    
    # Load environment variables
    if load_env_file():
        print("✅ Loaded environment variables from .env file")
    else:
        print("⚠️  No .env file found, using system environment variables")
    
    # Test connection
    if not test_agentdb_connection():
        print("\n❌ Connection test failed")
        print("\nPlease check your environment variables:")
        print("- AGENTDB_API_KEY")
        print("- AGENTDB_TOKEN")
        print("- AGENTDB_BASE_URL (optional)")
        print("- AGENTDB_DB_NAME (optional)")
        sys.exit(1)
    
    # Setup database
    if not setup_database():
        print("\n❌ Database setup failed")
        sys.exit(1)
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Run: python generate.py /path/to/your/images")
    print("2. Run: python index_agentdb.py")
    print("3. Run: streamlit run query_agentdb.py")

if __name__ == "__main__":
    main()