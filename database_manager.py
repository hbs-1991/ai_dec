import sqlite3
import json
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import os

class DatabaseManager:
    """
    Управление SQLite базой данных для хранения истории обработки
    и результатов классификации ТН ВЭД кодов
    """
    
    def __init__(self, db_path: str = "custbro_history.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Создает таблицы БД если они не существуют"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Таблица сессий обработки
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS processing_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    upload_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    total_items INTEGER DEFAULT 0,
                    processed_items INTEGER DEFAULT 0,
                    high_confidence_items INTEGER DEFAULT 0,
                    medium_confidence_items INTEGER DEFAULT 0,
                    low_confidence_items INTEGER DEFAULT 0,
                    status TEXT DEFAULT 'processing',
                    processing_time_seconds REAL DEFAULT 0.0,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Таблица результатов классификации
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS classification_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER NOT NULL,
                    row_index INTEGER NOT NULL,
                    product_name TEXT,
                    original_description TEXT,
                    category TEXT,
                    brand TEXT,
                    additional_info TEXT,
                    hs_code TEXT,
                    confidence_percentage INTEGER,
                    ai_reasoning TEXT,
                    alternative_codes TEXT,
                    user_status TEXT DEFAULT 'pending',
                    user_notes TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES processing_sessions (id)
                )
            """)
            
            conn.commit()
    
    def create_processing_session(self, filename: str, total_items: int) -> int:
        """
        Создает новую сессию обработки
        
        Returns:
            session_id: ID созданной сессии
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO processing_sessions (filename, total_items, status)
                VALUES (?, ?, 'processing')
            """, (filename, total_items))
            session_id = cursor.lastrowid
            conn.commit()
            return session_id
    
    def update_processing_session(
        self, 
        session_id: int,
        processed_items: int = None,
        high_confidence: int = None,
        medium_confidence: int = None, 
        low_confidence: int = None,
        processing_time: float = None,
        status: str = None
    ):
        """Обновляет данные сессии обработки"""
        updates = []
        params = []
        
        if processed_items is not None:
            updates.append("processed_items = ?")
            params.append(processed_items)
        if high_confidence is not None:
            updates.append("high_confidence_items = ?")
            params.append(high_confidence)
        if medium_confidence is not None:
            updates.append("medium_confidence_items = ?")
            params.append(medium_confidence)
        if low_confidence is not None:
            updates.append("low_confidence_items = ?")
            params.append(low_confidence)
        if processing_time is not None:
            updates.append("processing_time_seconds = ?")
            params.append(processing_time)
        if status is not None:
            updates.append("status = ?")
            params.append(status)
        
        if updates:
            params.append(session_id)
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(f"""
                    UPDATE processing_sessions 
                    SET {', '.join(updates)}
                    WHERE id = ?
                """, params)
                conn.commit()
    
    def save_classification_results(self, session_id: int, results: List[Dict]):
        """Сохраняет результаты классификации для сессии"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            for result in results:
                # Подготавливаем альтернативные коды как JSON
                alternative_codes = json.dumps(result.get('alternative_codes', []))
                
                cursor.execute("""
                    INSERT INTO classification_results (
                        session_id, row_index, product_name, original_description,
                        category, brand, additional_info, hs_code, 
                        confidence_percentage, ai_reasoning, alternative_codes
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    session_id,
                    result.get('row_index', 0),
                    result.get('product_name', ''),
                    result.get('original_description', ''),
                    result.get('category', ''),
                    result.get('brand', ''),
                    result.get('additional_info', ''),
                    result.get('hs_code', ''),
                    result.get('confidence_percentage', 0),
                    result.get('ai_reasoning', ''),
                    alternative_codes
                ))
            
            conn.commit()
    
    def get_recent_sessions(self, limit: int = 5) -> List[Dict]:
        """Получает список последних сессий обработки"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 
                    id, filename, upload_timestamp, total_items, 
                    processed_items, status, processing_time_seconds,
                    high_confidence_items, medium_confidence_items, low_confidence_items
                FROM processing_sessions 
                ORDER BY upload_timestamp DESC 
                LIMIT ?
            """, (limit,))
            
            sessions = []
            for row in cursor.fetchall():
                sessions.append({
                    'id': row[0],
                    'filename': row[1],
                    'upload_timestamp': row[2],
                    'total_items': row[3],
                    'processed_items': row[4],
                    'status': row[5],
                    'processing_time_seconds': row[6],
                    'high_confidence_items': row[7],
                    'medium_confidence_items': row[8],
                    'low_confidence_items': row[9]
                })
            
            return sessions
    
    def get_session_results(self, session_id: int) -> pd.DataFrame:
        """Получает результаты классификации для сессии в виде DataFrame"""
        with sqlite3.connect(self.db_path) as conn:
            query = """
                SELECT 
                    r.id, r.row_index, r.product_name, r.original_description,
                    r.category, r.brand, r.additional_info, r.hs_code,
                    r.confidence_percentage, r.ai_reasoning, r.alternative_codes,
                    r.user_status, r.user_notes, r.created_at
                FROM classification_results r
                WHERE r.session_id = ?
                ORDER BY r.row_index
            """
            
            df = pd.read_sql_query(query, conn, params=(session_id,))
            
            # Парсим JSON альтернативных кодов
            if not df.empty:
                df['alternative_codes'] = df['alternative_codes'].apply(
                    lambda x: json.loads(x) if x else []
                )
            
            return df
    
    def update_user_status(
        self, 
        result_id: int, 
        user_status: str, 
        user_notes: str = None
    ):
        """Обновляет пользовательский статус результата"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE classification_results 
                SET user_status = ?, user_notes = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (user_status, user_notes, result_id))
            conn.commit()
    
    def get_statistics(self) -> Dict:
        """Получает общую статистику по всем сессиям"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Общая статистика сессий
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_sessions,
                    SUM(total_items) as total_items_processed,
                    SUM(high_confidence_items) as total_high_confidence,
                    SUM(medium_confidence_items) as total_medium_confidence,
                    SUM(low_confidence_items) as total_low_confidence,
                    AVG(processing_time_seconds) as avg_processing_time
                FROM processing_sessions 
                WHERE status = 'completed'
            """)
            
            session_stats = cursor.fetchone()
            
            # Статистика пользовательских действий
            cursor.execute("""
                SELECT 
                    user_status,
                    COUNT(*) as count
                FROM classification_results
                GROUP BY user_status
            """)
            
            user_action_stats = {row[0]: row[1] for row in cursor.fetchall()}
            
            return {
                'total_sessions': session_stats[0] or 0,
                'total_items_processed': session_stats[1] or 0,
                'total_high_confidence': session_stats[2] or 0,
                'total_medium_confidence': session_stats[3] or 0,
                'total_low_confidence': session_stats[4] or 0,
                'avg_processing_time': session_stats[5] or 0.0,
                'user_actions': user_action_stats
            }
    
    def delete_session(self, session_id: int):
        """Удаляет сессию и все связанные результаты"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Удаляем результаты
            cursor.execute("DELETE FROM classification_results WHERE session_id = ?", (session_id,))
            
            # Удаляем сессию
            cursor.execute("DELETE FROM processing_sessions WHERE id = ?", (session_id,))
            
            conn.commit()
    
    def get_result_details(self, result_id: int) -> Dict:
        """Получает детальную информацию о результате классификации"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 
                    r.id, r.session_id, r.row_index, r.product_name, 
                    r.original_description, r.category, r.brand, r.additional_info,
                    r.hs_code, r.confidence_percentage, r.ai_reasoning, 
                    r.alternative_codes, r.user_status, r.user_notes,
                    r.created_at, r.updated_at,
                    s.filename
                FROM classification_results r
                JOIN processing_sessions s ON r.session_id = s.id
                WHERE r.id = ?
            """, (result_id,))
            
            row = cursor.fetchone()
            if row:
                return {
                    'id': row[0],
                    'session_id': row[1],
                    'row_index': row[2],
                    'product_name': row[3],
                    'original_description': row[4],
                    'category': row[5],
                    'brand': row[6],
                    'additional_info': row[7],
                    'hs_code': row[8],
                    'confidence_percentage': row[9],
                    'ai_reasoning': row[10],
                    'alternative_codes': json.loads(row[11]) if row[11] else [],
                    'user_status': row[12],
                    'user_notes': row[13],
                    'created_at': row[14],
                    'updated_at': row[15],
                    'filename': row[16]
                }
            return None

    def update_result_user_status(
        self, 
        result_id: int, 
        user_status: str, 
        user_notes: str = None
    ) -> bool:
        """
        Обновляет пользовательский статус и заметки для результата
        
        Args:
            result_id: ID результата классификации
            user_status: Новый статус ('pending', 'confirmed', 'needs_review', 'rejected')
            user_notes: Пользовательские заметки
            
        Returns:
            bool: True если обновление прошло успешно
        """
        valid_statuses = ['pending', 'confirmed', 'needs_review', 'rejected']
        if user_status not in valid_statuses:
            raise ValueError(f"Недопустимый статус: {user_status}. Разрешены: {valid_statuses}")
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE classification_results 
                    SET user_status = ?, user_notes = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (user_status, user_notes, result_id))
                
                # Проверяем, что строка была обновлена
                if cursor.rowcount > 0:
                    conn.commit()
                    return True
                return False
        except Exception as e:
            print(f"❌ Ошибка обновления статуса: {e}")
            return False

    def get_user_status_stats(self, session_id: int = None) -> Dict:
        """
        Получает статистику по пользовательским статусам
        
        Args:
            session_id: ID сессии (если None, то по всем сессиям)
            
        Returns:
            Dict со статистикой по статусам
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            if session_id:
                cursor.execute("""
                    SELECT 
                        user_status,
                        COUNT(*) as count
                    FROM classification_results 
                    WHERE session_id = ?
                    GROUP BY user_status
                """, (session_id,))
            else:
                cursor.execute("""
                    SELECT 
                        user_status,
                        COUNT(*) as count
                    FROM classification_results 
                    GROUP BY user_status
                """)
            
            stats = {
                'pending': 0,
                'confirmed': 0,
                'needs_review': 0,
                'rejected': 0
            }
            
            for row in cursor.fetchall():
                status, count = row
                if status in stats:
                    stats[status] = count
            
            return stats 