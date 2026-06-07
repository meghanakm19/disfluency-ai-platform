"""
Database Module
SQLite database for storing analysis results and user data
"""
import sqlite3
import json
import os
from datetime import datetime
from pathlib import Path
from uuid import uuid4


LABEL_TYPE_MAP = {
    'Filled Pause': 'FP',
    'Partial Repetition': 'RP',
    'Revision': 'RV',
    'Restart': 'RS',
    'Prolonged Word': 'PW',
    'Prolongation': 'PW',
}

class Database:
    """Database handler for analysis results"""
    
    def __init__(self, db_path):
        """Initialize database"""
        self.db_path = db_path
        self._shared_connection = db_path == ':memory:' or (db_path.startswith('file:') and 'mode=memory' in db_path)
        self._connection = None

        if self._shared_connection:
            self._connection = sqlite3.connect(
                db_path,
                check_same_thread=False,
                uri=db_path.startswith('file:')
            )
            self._connection.execute('PRAGMA foreign_keys = ON')
        else:
            directory = os.path.dirname(db_path)
            if directory:
                os.makedirs(directory, exist_ok=True)

    def _get_connection(self):
        """Get a connection with foreign key enforcement enabled."""
        if self._shared_connection:
            return self._connection

        conn = sqlite3.connect(self.db_path)
        conn.execute('PRAGMA foreign_keys = ON')
        return conn
    
    def init_db(self):
        """Initialize database tables"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                name TEXT,
                email TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_analysis TIMESTAMP
            )
        ''')
        
        # Analyses table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analyses (
                analysis_id TEXT PRIMARY KEY,
                user_id TEXT,
                filename TEXT,
                filepath TEXT,
                modality TEXT,
                results TEXT,
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            )
        ''')
        
        # Statistics cache table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS statistics (
                stat_id INTEGER PRIMARY KEY AUTOINCREMENT,
                stat_type TEXT,
                stat_value TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        if not self._shared_connection:
            conn.close()
    
    def save_analysis(self, user_id, filename, filepath, modality, results, notes=''):
        """Save analysis results to database"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Ensure user exists
            cursor.execute('''
                INSERT OR IGNORE INTO users (user_id, created_at)
                VALUES (?, ?)
            ''', (user_id, datetime.now()))
            
            # Generate analysis ID
            analysis_id = uuid4().hex
            
            # Save analysis
            cursor.execute('''
                INSERT INTO analyses 
                (analysis_id, user_id, filename, filepath, modality, results, notes, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                analysis_id,
                user_id,
                filename,
                filepath,
                modality,
                json.dumps(results),
                notes,
                datetime.now()
            ))
            
            # Update user's last_analysis timestamp
            cursor.execute('''
                UPDATE users SET last_analysis = ? WHERE user_id = ?
            ''', (datetime.now(), user_id))
            
            conn.commit()
            if not self._shared_connection:
                conn.close()
            
            return analysis_id
        
        except Exception as e:
            print(f"Error saving analysis: {str(e)}")
            raise
    
    def get_analysis(self, analysis_id):
        """Get analysis by ID"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT analysis_id, user_id, filename, filepath, modality, results, notes, created_at
                FROM analyses WHERE analysis_id = ?
            ''', (analysis_id,))
            
            row = cursor.fetchone()
            if not self._shared_connection:
                conn.close()
            
            if row:
                return {
                    'analysis_id': row[0],
                    'user_id': row[1],
                    'filename': row[2],
                    'filepath': row[3],
                    'modality': row[4],
                    'results': json.loads(row[5]),
                    'notes': row[6],
                    'created_at': row[7]
                }
            return None
        
        except Exception as e:
            print(f"Error getting analysis: {str(e)}")
            return None
    
    def get_user_analyses(self, user_id):
        """Get all analyses for a user"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT analysis_id, filename, modality, created_at
                FROM analyses 
                WHERE user_id = ?
                ORDER BY created_at DESC
            ''', (user_id,))
            
            rows = cursor.fetchall()
            if not self._shared_connection:
                conn.close()
            
            return [
                {
                    'analysis_id': row[0],
                    'filename': row[1],
                    'modality': row[2],
                    'created_at': row[3]
                }
                for row in rows
            ]
        
        except Exception as e:
            print(f"Error getting user analyses: {str(e)}")
            return []
    
    def get_all_analyses(self, page=1, per_page=20, user_id=None):
        """Get analyses with pagination, optionally filtered by user."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Get total count
            if user_id:
                cursor.execute('SELECT COUNT(*) FROM analyses WHERE user_id = ?', (user_id,))
            else:
                cursor.execute('SELECT COUNT(*) FROM analyses')
            total = cursor.fetchone()[0]
            
            # Get paginated results
            offset = (page - 1) * per_page
            if user_id:
                cursor.execute('''
                    SELECT analysis_id, user_id, filename, modality, created_at
                    FROM analyses
                    WHERE user_id = ?
                    ORDER BY created_at DESC
                    LIMIT ? OFFSET ?
                ''', (user_id, per_page, offset))
            else:
                cursor.execute('''
                    SELECT analysis_id, user_id, filename, modality, created_at
                    FROM analyses
                    ORDER BY created_at DESC
                    LIMIT ? OFFSET ?
                ''', (per_page, offset))
            
            rows = cursor.fetchall()
            if not self._shared_connection:
                conn.close()
            
            return {
                'total': total,
                'page': page,
                'per_page': per_page,
                'user_id': user_id,
                'analyses': [
                    {
                        'analysis_id': row[0],
                        'user_id': row[1],
                        'filename': row[2],
                        'modality': row[3],
                        'created_at': row[4]
                    }
                    for row in rows
                ]
            }
        
        except Exception as e:
            print(f"Error getting all analyses: {str(e)}")
            return {'total': 0, 'analyses': []}
    
    def delete_analysis(self, analysis_id):
        """Delete an analysis"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Get filepath first
            cursor.execute('SELECT filepath FROM analyses WHERE analysis_id = ?', (analysis_id,))
            row = cursor.fetchone()
            
            if row and os.path.exists(row[0]):
                os.remove(row[0])
            
            # Delete from database
            cursor.execute('DELETE FROM analyses WHERE analysis_id = ?', (analysis_id,))
            conn.commit()
            if not self._shared_connection:
                conn.close()
            
            return cursor.rowcount > 0
        
        except Exception as e:
            print(f"Error deleting analysis: {str(e)}")
            return False
    
    def get_statistics(self):
        """Get system statistics"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Total analyses
            cursor.execute('SELECT COUNT(*) FROM analyses')
            total_analyses = cursor.fetchone()[0]
            
            # Total users
            cursor.execute('SELECT COUNT(*) FROM users')
            total_users = cursor.fetchone()[0]
            
            # Analyses by modality
            cursor.execute('''
                SELECT modality, COUNT(*) FROM analyses GROUP BY modality
            ''')
            modality_stats = {row[0]: row[1] for row in cursor.fetchall()}
            
            # Average disfluencies
            cursor.execute('SELECT results FROM analyses')
            results = cursor.fetchall()
            
            total_disfluencies = 0
            total_disfluency_rate = 0.0
            rate_samples = 0
            label_counts = {
                'FP': 0,
                'RP': 0,
                'RV': 0,
                'RS': 0,
                'PW': 0
            }
            if results:
                for row in results:
                    try:
                        data = json.loads(row[0])
                        predictions = data.get('predictions') or []
                        if predictions:
                            disfluent_frames = sum(1 for prediction in predictions if prediction.get('is_disfluent'))
                            total_frames = len(predictions)
                            total_disfluencies += sum(
                                1 for index, prediction in enumerate(predictions)
                                if prediction.get('is_disfluent')
                                and (index == 0 or not predictions[index - 1].get('is_disfluent'))
                            )
                            total_disfluency_rate += (disfluent_frames / total_frames) * 100 if total_frames else 0.0
                            rate_samples += 1

                            for prediction in predictions:
                                if not prediction.get('is_disfluent'):
                                    continue
                                for label_name in prediction.get('disfluency_types', []):
                                    mapped_label = LABEL_TYPE_MAP.get(label_name)
                                    if mapped_label in label_counts:
                                        label_counts[mapped_label] += 1
                        else:
                            if 'disfluency_count' in data:
                                total_disfluencies += int(data['disfluency_count'])
                            if isinstance(data.get('statistics'), dict) and 'disfluency_rate' in data['statistics']:
                                total_disfluency_rate += float(data['statistics']['disfluency_rate'])
                                rate_samples += 1
                            for label in label_counts:
                                if f'{label}_count' in data:
                                    label_counts[label] += int(data[f'{label}_count'])
                    except:
                        pass
            
            if not self._shared_connection:
                conn.close()
            
            return {
                'total_analyses': total_analyses,
                'total_users': total_users,
                'analyses_by_modality': modality_stats,
                'total_disfluencies_detected': total_disfluencies,
                'disfluencies_by_label': label_counts,
                'avg_disfluencies': round(total_disfluencies / total_analyses, 2) if total_analyses > 0 else 0,
                'avg_disfluency_rate': round(total_disfluency_rate / rate_samples, 2) if rate_samples > 0 else 0
            }
        
        except Exception as e:
            print(f"Error getting statistics: {str(e)}")
            return {}
