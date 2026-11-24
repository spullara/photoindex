"""
AgentDB Python SDK

A lightweight Python SDK for the AgentDB database service.
Uses the requests library for HTTP requests.
"""

import json
import requests
from typing import List, Dict, Any, Optional, Union
from urllib.parse import urlencode


class AgentDBError(Exception):
    """Base exception for AgentDB operations"""
    def __init__(self, message: str, status_code: Optional[int] = None, response: Optional[Dict] = None):
        super().__init__(message)
        self.status_code = status_code
        self.response = response


class AuthenticationError(AgentDBError):
    """Authentication error for AgentDB operations"""
    def __init__(self, message: str, status_code: int = 401, response: Optional[Dict] = None):
        super().__init__(message, status_code, response)


class ValidationError(AgentDBError):
    """Validation error for AgentDB operations"""
    def __init__(self, message: str, status_code: int = 400, response: Optional[Dict] = None):
        super().__init__(message, status_code, response)


class DatabaseError(AgentDBError):
    """Database error for AgentDB operations"""
    def __init__(self, message: str, status_code: int = 500, response: Optional[Dict] = None):
        super().__init__(message, status_code, response)


class DatabaseService:
    """Main database service class for listing databases and creating connections"""
    
    def __init__(self, base_url: str, api_key: str, debug: bool = False):
        """
        Create a new DatabaseService instance
        
        Args:
            base_url: The base URL of the AgentDB service (e.g., 'https://api.agentdb.dev')
            api_key: The API key for authentication
            debug: Enable debug logging (default: False)
        """
        if not base_url:
            raise ValidationError('base_url is required')
        if not api_key:
            raise ValidationError('api_key is required')
        
        self.base_url = base_url.rstrip('/')  # Remove trailing slash
        self.api_key = api_key
        self.debug = debug
    
    def _debug_log(self, *args):
        """Log debug messages when debug mode is enabled"""
        if self.debug:
            print('[AgentDB SDK Debug]', *args)
    
    def _make_request(self, endpoint: str, method: str = 'GET', data: Optional[Dict] = None, params: Optional[Dict] = None) -> Dict:
        """Make an authenticated HTTP request"""
        url = f"{self.base_url}{endpoint}"
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        # Add debug parameter if debug mode is enabled
        if self.debug:
            if method == 'GET':
                if params is None:
                    params = {}
                params['debug'] = 'true'
            else:
                if data is None:
                    data = {}
                data['debug'] = True
        
        self._debug_log(f'Making {method} request to: {url}')
        if data:
            self._debug_log('Request body:', json.dumps(data, indent=2))
        
        try:
            response = requests.request(
                method=method,
                url=url,
                headers=headers,
                json=data,
                params=params
            )
            
            try:
                response_data = response.json()
            except json.JSONDecodeError as e:
                raise AgentDBError(f'Failed to parse response: {e}', response.status_code)
            
            self._debug_log('Response status:', response.status_code)
            self._debug_log('Response data:', json.dumps(response_data, indent=2))
            
            # Log debug information from API if available
            if response_data.get('debugLogs') and isinstance(response_data['debugLogs'], list):
                self._debug_log('API Debug Logs:')
                for log in response_data['debugLogs']:
                    self._debug_log(f"[{log.get('level')}] {log.get('timestamp')}: {log.get('message')}")
            
            if not response.ok:
                error_message = response_data.get('error', f'HTTP {response.status_code}: {response.reason}')
                
                if response.status_code == 401:
                    raise AuthenticationError(error_message, response.status_code, response_data)
                elif response.status_code == 400:
                    raise ValidationError(error_message, response.status_code, response_data)
                elif response.status_code >= 500:
                    raise DatabaseError(error_message, response.status_code, response_data)
                else:
                    raise AgentDBError(error_message, response.status_code, response_data)
            
            return response_data
        
        except requests.exceptions.RequestException as e:
            raise AgentDBError(f'Network error: {e}')
    
    def list_databases(self, token: str) -> List[Dict]:
        """
        List all databases for a given token
        
        Args:
            token: The UUID token for database access
            
        Returns:
            Array of database objects with name, type, fileName, size, and modified properties
        """
        if not token:
            raise ValidationError('token is required')
        
        response = self._make_request(
            f'/db/api/list-databases',
            method='GET',
            params={'token': token}
        )
        return response.get('databases', [])
    
    def connect(self, token: str, db_name: str, db_type: str = 'sqlite') -> 'DatabaseConnection':
        """
        Create a new database connection
        
        Args:
            token: The UUID token for database access
            db_name: The name of the database
            db_type: The type of database ('sqlite' or 'duckdb')
            
        Returns:
            A new database connection instance
        """
        return DatabaseConnection(self.base_url, self.api_key, token, db_name, db_type, self.debug)
    
    def delete_database(self, token: str, db_name: str, db_type: str = 'sqlite') -> Dict:
        """
        Delete a database
        
        Args:
            token: The UUID token for database access
            db_name: The name of the database to delete
            db_type: The type of database ('sqlite' or 'duckdb')
            
        Returns:
            Deletion result with success message and optional debug logs
        """
        if not token:
            raise ValidationError('token is required')
        if not db_name:
            raise ValidationError('db_name is required')
        if db_type not in ['sqlite', 'duckdb']:
            raise ValidationError('db_type must be either "sqlite" or "duckdb"')
        
        return self._make_request(
            '/db/api/delete-database',
            method='DELETE',
            data={'token': token, 'dbName': db_name, 'dbType': db_type}
        )
    
    def rename_database(self, token: str, old_db_name: str, new_db_name: str, db_type: str = 'sqlite') -> Dict:
        """
        Rename a database
        
        Args:
            token: The UUID token for database access
            old_db_name: The current name of the database
            new_db_name: The new name for the database
            db_type: The type of database ('sqlite' or 'duckdb')
            
        Returns:
            Rename result with success message and optional debug logs
        """
        if not token:
            raise ValidationError('token is required')
        if not old_db_name:
            raise ValidationError('old_db_name is required')
        if not new_db_name:
            raise ValidationError('new_db_name is required')
        if db_type not in ['sqlite', 'duckdb']:
            raise ValidationError('db_type must be either "sqlite" or "duckdb"')
        
        return self._make_request(
            '/db/api/rename-database',
            method='PUT',
            data={'token': token, 'oldDbName': old_db_name, 'newDbName': new_db_name, 'dbType': db_type}
        )
    
    def copy_database(self, source_token: str, source_db_name: str, source_db_type: str, 
                     dest_token: str, dest_db_name: str) -> Dict:
        """
        Copy a database from one location to another
        
        Args:
            source_token: The UUID token for the source database
            source_db_name: The name of the source database
            source_db_type: The type of the source database ('sqlite' or 'duckdb')
            dest_token: The UUID token for the destination
            dest_db_name: The name for the destination database
            
        Returns:
            Copy result with success message and optional debug logs
        """
        if not source_token:
            raise ValidationError('source_token is required')
        if not source_db_name:
            raise ValidationError('source_db_name is required')
        if not source_db_type:
            raise ValidationError('source_db_type is required')
        if not dest_token:
            raise ValidationError('dest_token is required')
        if not dest_db_name:
            raise ValidationError('dest_db_name is required')
        if source_db_type not in ['sqlite', 'duckdb']:
            raise ValidationError('source_db_type must be either "sqlite" or "duckdb"')
        
        return self._make_request(
            '/db/api/copy-database',
            method='POST',
            data={
                'sourceToken': source_token,
                'sourceDbName': source_db_name,
                'sourceDbType': source_db_type,
                'destToken': dest_token,
                'destDbName': dest_db_name
            }
        )
    
    def get_upload_url(self, token: str, db_name: str, db_type: str = 'sqlite') -> Dict:
        """
        Get a presigned URL for uploading a database file
        
        Args:
            token: The UUID token for database access
            db_name: The name for the database
            db_type: The type of database ('sqlite' or 'duckdb')
            
        Returns:
            Upload URL information with uploadUrl, fileName, expiresIn, and optional debug logs
        """
        if not token:
            raise ValidationError('token is required')
        if not db_name:
            raise ValidationError('db_name is required')
        if db_type not in ['sqlite', 'duckdb']:
            raise ValidationError('db_type must be either "sqlite" or "duckdb"')
        
        return self._make_request(
            '/db/api/upload-url',
            method='POST',
            data={'token': token, 'dbName': db_name, 'dbType': db_type}
        )
    
    def get_download_url(self, token: str, db_name: str, db_type: str = 'sqlite') -> Dict:
        """
        Get a presigned URL for downloading a database file
        
        Args:
            token: The UUID token for database access
            db_name: The name of the database
            db_type: The type of database ('sqlite' or 'duckdb')
            
        Returns:
            Download URL information with downloadUrl, fileName, expiresIn, and optional debug logs
        """
        if not token:
            raise ValidationError('token is required')
        if not db_name:
            raise ValidationError('db_name is required')
        if db_type not in ['sqlite', 'duckdb']:
            raise ValidationError('db_type must be either "sqlite" or "duckdb"')
        
        return self._make_request(
            '/db/api/download-url',
            method='POST',
            data={'token': token, 'dbName': db_name, 'dbType': db_type}
        )


class DatabaseConnection:
    """Database connection class for executing SQL statements"""
    
    def __init__(self, base_url: str, api_key: str, token: str, db_name: str, db_type: str = 'sqlite', debug: bool = False):
        """
        Create a new DatabaseConnection instance
        
        Args:
            base_url: The base URL of the AgentDB service
            api_key: The API key for authentication
            token: The UUID token for database access
            db_name: The name of the database
            db_type: The type of database ('sqlite' or 'duckdb')
            debug: Enable debug logging (default: False)
        """
        if not base_url:
            raise ValidationError('base_url is required')
        if not api_key:
            raise ValidationError('api_key is required')
        if not token:
            raise ValidationError('token is required')
        if not db_name:
            raise ValidationError('db_name is required')
        if db_type not in ['sqlite', 'duckdb']:
            raise ValidationError('db_type must be either "sqlite" or "duckdb"')
        
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.token = token
        self.db_name = db_name
        self.db_type = db_type
        self.debug = debug
    
    def _debug_log(self, *args):
        """Log debug messages when debug mode is enabled"""
        if self.debug:
            print('[AgentDB SDK Debug]', *args)
    
    def _make_request(self, endpoint: str, method: str = 'GET', data: Optional[Dict] = None) -> Dict:
        """Make an authenticated HTTP request"""
        url = f"{self.base_url}{endpoint}"
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        self._debug_log(f'Making {method} request to: {url}')
        if data:
            self._debug_log('Request body:', json.dumps(data, indent=2))
        
        try:
            response = requests.request(
                method=method,
                url=url,
                headers=headers,
                json=data
            )
            
            try:
                response_data = response.json()
            except json.JSONDecodeError as e:
                raise AgentDBError(f'Failed to parse response: {e}', response.status_code)
            
            self._debug_log('Response status:', response.status_code)
            self._debug_log('Response data:', json.dumps(response_data, indent=2))
            
            # Log debug information from API if available
            if response_data.get('debugLogs') and isinstance(response_data['debugLogs'], list):
                self._debug_log('API Debug Logs:')
                for log in response_data['debugLogs']:
                    self._debug_log(f"[{log.get('level')}] {log.get('timestamp')}: {log.get('message')}")
            
            if not response.ok:
                error_message = response_data.get('error', f'HTTP {response.status_code}: {response.reason}')
                
                if response.status_code == 401:
                    raise AuthenticationError(error_message, response.status_code, response_data)
                elif response.status_code == 400:
                    raise ValidationError(error_message, response.status_code, response_data)
                elif response.status_code >= 500:
                    raise DatabaseError(error_message, response.status_code, response_data)
                else:
                    raise AgentDBError(error_message, response.status_code, response_data)
            
            return response_data
        
        except requests.exceptions.RequestException as e:
            raise AgentDBError(f'Network error: {e}')
    
    def execute(self, statements: Union[Dict, List[Dict]]) -> Dict:
        """
        Execute one or more SQL statements
        
        Args:
            statements: Array of statement objects or single statement object
                       Each statement should have 'sql' and optional 'params' properties
                       
        Returns:
            Execution results with results array and optional debug logs
        """
        if not statements:
            raise ValidationError('statements is required')
        
        # Normalize to array format
        statements_array = statements if isinstance(statements, list) else [statements]
        
        # Validate statements format
        for stmt in statements_array:
            if not isinstance(stmt, dict):
                raise ValidationError('Each statement must be a dictionary')
            if not stmt.get('sql') or not isinstance(stmt['sql'], str):
                raise ValidationError('Each statement must have a "sql" property with a string value')
            if 'params' in stmt and not isinstance(stmt['params'], list):
                raise ValidationError('Statement params must be a list if provided')
        
        request_data = {
            'token': self.token,
            'dbName': self.db_name,
            'dbType': self.db_type,
            'statements': statements_array
        }
        
        # Add debug parameter if debug mode is enabled
        if self.debug:
            request_data['debug'] = True
        
        return self._make_request('/db/api/execute', method='POST', data=request_data)