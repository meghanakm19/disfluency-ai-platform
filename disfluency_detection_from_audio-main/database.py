"""
Database Module
SQLite database for storing analysis results and user data
"""
import sqlite3
import json
import os
import uuid
from datetime import datetime
from pathlib import Path

class Database:
    """Database handler for analysis results"""
    
    def __init__(self, db_path):
        """Initialize database"""
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    def init_db(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
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
        conn.close()
    
    def save_analysis(self, user_id, filename, filepath, modality, results, notes=''):
        """Save analysis results to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Ensure user exists
            cursor.execute('''
                INSERT OR IGNORE INTO users (user_id, created_at)
                VALUES (?, ?)
            ''', (user_id, datetime.now()))
            
            # Generate analysis ID with UUID to ensure uniqueness
            analysis_id = f"{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
            
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
            conn.close()
            
            return analysis_id
        
        except Exception as e:
            print(f"Error saving analysis: {str(e)}")
            raise
    
    def get_analysis(self, analysis_id):
        """Get analysis by ID"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT analysis_id, user_id, filename, filepath, modality, results, notes, created_at
                FROM analyses WHERE analysis_id = ?
            ''', (analysis_id,))
            
            row = cursor.fetchone()
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
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT analysis_id, filename, modality, created_at
                FROM analyses 
                WHERE user_id = ?
                ORDER BY created_at DESC
            ''', (user_id,))
            
            rows = cursor.fetchall()
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
        """Get all analyses with pagination, optionally filtered by user_id"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Build query based on whether user_id filter is provided
            if user_id:
                cursor.execute('SELECT COUNT(*) FROM analyses WHERE user_id = ?', (user_id,))
                total = cursor.fetchone()[0]
                
                offset = (page - 1) * per_page
                cursor.execute('''
                    SELECT analysis_id, user_id, filename, modality, created_at
                    FROM analyses
                    WHERE user_id = ?
                    ORDER BY created_at DESC
                    LIMIT ? OFFSET ?
                ''', (user_id, per_page, offset))
            else:
                cursor.execute('SELECT COUNT(*) FROM analyses')
                total = cursor.fetchone()[0]
                
                offset = (page - 1) * per_page
                cursor.execute('''
                    SELECT analysis_id, user_id, filename, modality, created_at
                    FROM analyses
                    ORDER BY created_at DESC
                    LIMIT ? OFFSET ?
                ''', (per_page, offset))
            
            rows = cursor.fetchall()
            conn.close()
            
            return {
                'total': total,
                'page': page,
                'per_page': per_page,
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
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get filepath first
            cursor.execute('SELECT filepath FROM analyses WHERE analysis_id = ?', (analysis_id,))
            row = cursor.fetchone()
            
            if row and os.path.exists(row[0]):
                os.remove(row[0])
            
            # Delete from database
            cursor.execute('DELETE FROM analyses WHERE analysis_id = ?', (analysis_id,))
            conn.commit()
            conn.close()
            
            return cursor.rowcount > 0
        
        except Exception as e:
            print(f"Error deleting analysis: {str(e)}")
            return False
    
    def get_statistics(self):
        """Get system statistics"""
        try:
            conn = sqlite3.connect(self.db_path)
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
            if results:
                for row in results:
                    try:
                        data = json.loads(row[0])
                        if 'disfluency_count' in data:
                            total_disfluencies += data['disfluency_count']
                    except:
                        pass
            
            conn.close()
            
            return {
                'total_analyses': total_analyses,
                'total_users': total_users,
                'analyses_by_modality': modality_stats,
                'total_disfluencies_detected': total_disfluencies,
                'avg_disfluencies': round(total_disfluencies / total_analyses, 2) if total_analyses > 0 else 0
            }
        
        except Exception as e:
            print(f"Error getting statistics: {str(e)}")
            return {}
