"""
AI DECLARANT - Data Processor
Модуль для обработки данных и интеграции с AI агентом
"""

import asyncio
import pandas as pd
import streamlit as st
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime
import io
import config
import numpy as np
from ai_agents.hs_agent import HSCodeAgent
from database_manager import DatabaseManager
import time

from ai_agents.hs_agent import (
    HSCodeAgent, 
    HSCodeSearchResult, 
    BatchProcessingResult
)


class DataProcessor:
    """Основной класс для обработки данных"""
    
    def __init__(self):
        """Инициализирует процессор данных"""
        self.current_data = None
        self.column_mapping = {}
        self.agent = None
        self.db_manager = DatabaseManager()  # Инициализируем менеджер БД
        print("📊 DataProcessor инициализирован с поддержкой БД")
        self.hs_agent = HSCodeAgent()
        self.supported_formats = config.SUPPORTED_FILE_TYPES
        self.max_rows = config.MAX_ROWS_PER_FILE
    
    def validate_file(self, uploaded_file) -> Dict[str, Any]:
        """
        Валидирует загруженный файл
        
        Args:
            uploaded_file: Загруженный файл из Streamlit
            
        Returns:
            Словарь с результатами валидации
        """
        validation_result = {
            "is_valid": False,
            "error_message": "",
            "file_info": {},
            "warnings": []
        }
        
        try:
            # Проверяем размер файла
            file_size_mb = uploaded_file.size / (1024 * 1024)
            if file_size_mb > config.MAX_FILE_SIZE_MB:
                validation_result["error_message"] = f"Файл слишком большой ({file_size_mb:.1f} MB). Максимальный размер: {config.MAX_FILE_SIZE_MB} MB"
                return validation_result
            
            # Проверяем расширение файла
            file_extension = uploaded_file.name.split('.')[-1].lower()
            if file_extension not in self.supported_formats:
                validation_result["error_message"] = f"Неподдерживаемый формат файла: .{file_extension}. Поддерживаемые форматы: {', '.join(self.supported_formats)}"
                return validation_result
            
            # Собираем информацию о файле
            validation_result["file_info"] = {
                "name": uploaded_file.name,
                "size_mb": round(file_size_mb, 2),
                "format": file_extension.upper(),
                "upload_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            validation_result["is_valid"] = True
            
        except Exception as e:
            validation_result["error_message"] = f"Ошибка при валидации файла: {str(e)}"
        
        return validation_result
    
    def load_data(self, uploaded_file) -> Dict[str, Any]:
        """
        Загружает данные из файла
        
        Args:
            uploaded_file: Загруженный файл из Streamlit
            
        Returns:
            Словарь с данными и метаинформацией
        """
        result = {
            "success": False,
            "data": None,
            "error_message": "",
            "file_info": {},
            "sample_data": None
        }
        
        try:
            # Валидируем файл
            validation = self.validate_file(uploaded_file)
            if not validation["is_valid"]:
                result["error_message"] = validation["error_message"]
                return result
            
            result["file_info"] = validation["file_info"]
            
            # Читаем данные в зависимости от формата
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            if file_extension == 'csv':
                # Пробуем различные кодировки для CSV
                encodings = ['utf-8', 'windows-1251', 'cp1252', 'iso-8859-1']
                data = None
                
                for encoding in encodings:
                    try:
                        uploaded_file.seek(0)  # Сбрасываем указатель файла
                        data = pd.read_csv(uploaded_file, encoding=encoding)
                        break
                    except UnicodeDecodeError:
                        continue
                
                if data is None:
                    result["error_message"] = "Не удалось определить кодировку CSV файла"
                    return result
                    
            elif file_extension in ['xlsx', 'xls']:
                try:
                    uploaded_file.seek(0)
                    data = pd.read_excel(uploaded_file)
                except Exception as e:
                    result["error_message"] = f"Ошибка чтения Excel файла: {str(e)}"
                    return result
            
            # Проверяем количество строк
            if len(data) > self.max_rows:
                result["error_message"] = f"Слишком много строк ({len(data)}). Максимум: {self.max_rows}"
                return result
            
            # Проверяем, что данные не пустые
            if data.empty:
                result["error_message"] = "Файл не содержит данных"
                return result
            
            # Очищаем данные
            data = self._clean_data(data)
            
            # Создаем образец данных для предпросмотра
            sample_size = min(10, len(data))
            sample_data = data.head(sample_size)
            
            result.update({
                "success": True,
                "data": data,
                "sample_data": sample_data,
                "file_info": {
                    **result["file_info"],
                    "rows_count": len(data),
                    "columns_count": len(data.columns),
                    "columns": list(data.columns)
                }
            })
            
        except Exception as e:
            result["error_message"] = f"Неожиданная ошибка при загрузке данных: {str(e)}"
        
        return result
    
    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Очищает данные от пустых строк и лишних пробелов
        
        Args:
            data: Исходный DataFrame
            
        Returns:
            Очищенный DataFrame
        """
        # Удаляем полностью пустые строки
        data = data.dropna(how='all')
        
        # Обрезаем пробелы в текстовых колонках
        text_columns = data.select_dtypes(include=['object']).columns
        data[text_columns] = data[text_columns].apply(lambda x: x.str.strip() if x.dtype == "object" else x)
        
        # Заменяем NaN на пустые строки для текстовых полей
        data[text_columns] = data[text_columns].fillna('')
        
        return data
    
    def map_columns(self, data: pd.DataFrame, column_mapping: Dict[str, str]) -> pd.DataFrame:
        """
        Маппит колонки согласно пользовательскому выбору
        
        Args:
            data: Исходные данные
            column_mapping: Маппинг колонок {исходная_колонка: целевая_колонка}
            
        Returns:
            DataFrame с переименованными колонками
        """
        # Создаем копию данных
        mapped_data = data.copy()
        
        # Переименовываем колонки согласно маппингу
        reverse_mapping = {v: k for k, v in column_mapping.items() if v and v != "Не использовать"}
        
        if reverse_mapping:
            mapped_data = mapped_data.rename(columns=reverse_mapping)
        
        return mapped_data
    
    def prepare_items_for_processing(self, data: pd.DataFrame, column_mapping: Dict[str, str]) -> List[Dict[str, str]]:
        """
        Подготавливает элементы для обработки AI агентом
        
        Args:
            data: DataFrame с данными
            column_mapping: Сопоставление колонок
            
        Returns:
            Список словарей с данными для AI агента
        """
        print(f"🔧 [DEBUG] prepare_items_for_processing: обрабатываем {len(data)} строк")
        print(f"🗂️ [DEBUG] Сопоставление колонок: {column_mapping}")
        
        items = []
        
        # Определяем главную колонку с названием товара
        main_column = None
        for target, source in column_mapping.items():
            if target == "product_name" and source:
                main_column = source
                break
        
        if not main_column:
            print("❌ [DEBUG] Главная колонка с названием товара не найдена!")
            raise ValueError("Не удалось определить главную колонку с названием товара")
        
        print(f"📝 [DEBUG] Главная колонка: '{main_column}'")
        
        # Создаем элементы для обработки
        for index, row in data.iterrows():
            # Безопасно получаем значение главной колонки
            main_value = row.get(main_column, '')
            if pd.isna(main_value):
                main_value = ''
            product_name = str(main_value).strip()
            
            if not product_name:
                print(f"⚠️ [DEBUG] Строка {index}: пустое название товара, пропускаем")
                continue
            
            item = {
                "row_index": index,
                "product_name": product_name
            }
            
            # Добавляем дополнительную информацию из других колонок
            for target, source in column_mapping.items():
                if source and source != "Не использовать" and source != main_column and source in data.columns:
                    raw_value = row.get(source, '')
                    if pd.isna(raw_value):
                        raw_value = ''
                    value = str(raw_value).strip()
                    if value:
                        item[target] = value
            
            # Добавляем остальные колонки как дополнительную информацию
            for col in data.columns:
                if col not in column_mapping.values() and col != main_column:
                    raw_value = row.get(col, '')
                    if pd.isna(raw_value):
                        raw_value = ''
                    value = str(raw_value).strip()
                    if value and value.lower() not in ['nan', 'none', 'null']:
                        # Используем имя колонки как ключ
                        clean_key = col.lower().replace(' ', '_').replace('-', '_')
                        item[clean_key] = value
            
            print(f"  ✅ [DEBUG] Элемент {len(items) + 1}: {item}")
            items.append(item)
        
        print(f"🎯 [DEBUG] Подготовлено {len(items)} элементов для обработки AI агентом")
        return items
    
    async def process_data_with_ai(
        self, 
        data: pd.DataFrame, 
        column_mapping: Dict[str, str],
        filename: str = "unknown_file",
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Обрабатывает данные с помощью AI агента с сохранением в БД
        
        Args:
            data: DataFrame с данными для обработки
            column_mapping: Сопоставление колонок
            filename: Имя файла для истории
            progress_callback: Функция для отслеживания прогресса
            
        Returns:
            Словарь с результатами обработки и session_id
        """
        print(f"🚀 [DEBUG] Начинаем обработку файла: {filename}")
        start_time = time.time()
        
        # Создаем сессию в БД
        session_id = self.db_manager.create_processing_session(
            filename=filename,
            total_items=len(data)
        )
        print(f"📊 [DEBUG] Создана сессия БД: {session_id}")
        
        try:
            # Подготавливаем элементы для обработки
            items = self.prepare_items_for_processing(data, column_mapping)
            
            if not items:
                print("❌ [ERROR] Нет элементов для обработки")
                self.db_manager.update_processing_session(
                    session_id=session_id,
                    status='failed'
                )
                return {"success": False, "error": "Нет данных для обработки", "session_id": session_id}
            
            # Обрабатываем с помощью AI агента
            batch_result = await self.hs_agent.classify_batch(items, progress_callback)
            
            # Проверяем успешность обработки (BatchProcessingResult не имеет success)
            if not batch_result.results or len(batch_result.errors) > 0:
                error_msg = f"Ошибка AI агента: {'; '.join(batch_result.errors)}" if batch_result.errors else "Нет результатов обработки"
                print(f"❌ [ERROR] {error_msg}")
                self.db_manager.update_processing_session(
                    session_id=session_id,
                    status='failed'
                )
                return {
                    "success": False, 
                    "error": error_msg,
                    "session_id": session_id
                }
            
            # Создаем DataFrame с результатами
            results_df = self.create_results_dataframe(data, batch_result)
            
            # Подсчитываем статистику из processing_stats
            stats = batch_result.processing_stats
            confidence_stats = {
                'high': stats.get('high_confidence_results', 0),
                'medium': stats.get('successful_classifications', 0) - stats.get('high_confidence_results', 0),
                'low': stats.get('total_items', 0) - stats.get('successful_classifications', 0)
            }
            processing_time = time.time() - start_time
            
            # Обновляем сессию в БД
            self.db_manager.update_processing_session(
                session_id=session_id,
                processed_items=len(batch_result.results),
                high_confidence=confidence_stats['high'],
                medium_confidence=confidence_stats['medium'],
                low_confidence=confidence_stats['low'],
                processing_time=processing_time,
                status='completed'
            )
            
            # Подготавливаем результаты для сохранения в БД
            db_results = []
            for i, ai_result in enumerate(batch_result.results):
                row_index = items[i].get('row_index', i) if i < len(items) else i
                original_row = data.iloc[row_index] if row_index < len(data) else pd.Series()
                
                # Получаем дополнительную информацию из исходных данных
                category = brand = additional_info = ''
                
                for target, source in column_mapping.items():
                    if source and source in original_row.index:
                        value = str(original_row.get(source, '')).strip()
                        if target == 'category':
                            category = value
                        elif target == 'brand':
                            brand = value
                        elif target == 'additional_info':
                            additional_info = value
                
                db_result = {
                    'row_index': row_index,
                    'product_name': items[i].get('product_name', ''),
                    'original_description': ai_result.description or '',
                    'category': category,
                    'brand': brand,
                    'additional_info': additional_info,
                    'hs_code': ai_result.hs_code,
                    'confidence_percentage': ai_result.confidence,
                    'ai_reasoning': ai_result.reasoning,
                    'alternative_codes': getattr(ai_result, 'alternative_codes', [])
                }
                
                db_results.append(db_result)
            
            # Сохраняем результаты в БД
            self.db_manager.save_classification_results(session_id, db_results)
            
            print(f"✅ [SUCCESS] Обработка завершена за {processing_time:.2f}с, сохранено в БД")
            
            return {
                "success": True,
                "results": [
                    {
                        'row_index': i,
                        'product_name': items[i].get('product_name', ''),
                        'hs_code': result.hs_code,
                        'confidence_percentage': result.confidence,
                        'reasoning': result.reasoning,
                        'alternative_codes': getattr(result, 'alternative_codes', [])
                    }
                    for i, result in enumerate(batch_result.results)
                ],
                "dataframe": results_df,
                "statistics": confidence_stats,
                "processing_time": processing_time,
                "session_id": session_id
            }
            
        except Exception as e:
            print(f"❌ [EXCEPTION] Критическая ошибка: {str(e)}")
            self.db_manager.update_processing_session(
                session_id=session_id,
                status='failed'
            )
            return {
                "success": False, 
                "error": str(e), 
                "session_id": session_id
            }
    
    def calculate_confidence_stats(self, results: List[Dict]) -> Dict[str, int]:
        """Подсчитывает статистику по уровням уверенности"""
        stats = {'high': 0, 'medium': 0, 'low': 0}
        
        for result in results:
            confidence = result.get('confidence_percentage', 0)
            if confidence >= 80:
                stats['high'] += 1
            elif confidence >= 40:
                stats['medium'] += 1
            else:
                stats['low'] += 1
        
        return stats
    
    def prepare_results_for_db(
        self, 
        original_data: pd.DataFrame, 
        ai_results: List[Dict], 
        column_mapping: Dict[str, str]
    ) -> List[Dict]:
        """Подготавливает результаты для сохранения в БД"""
        db_results = []
        
        for ai_result in ai_results:
            row_index = ai_result.get('row_index', 0)
            original_row = original_data.iloc[row_index] if row_index < len(original_data) else pd.Series()
            
            # Получаем дополнительную информацию из исходных данных
            category = ''
            brand = ''
            additional_info = ''
            
            for target, source in column_mapping.items():
                if source and source in original_data.columns:
                    value = str(original_row.get(source, '')).strip()
                    if target == 'category':
                        category = value
                    elif target == 'brand':
                        brand = value
                    elif target == 'additional_info':
                        additional_info = value
            
            db_result = {
                'row_index': row_index,
                'product_name': ai_result.get('product_name', ''),
                'original_description': ai_result.get('original_description', ''),
                'category': category,
                'brand': brand,
                'additional_info': additional_info,
                'hs_code': ai_result.get('hs_code', ''),
                'confidence_percentage': ai_result.get('confidence_percentage', 0),
                'ai_reasoning': ai_result.get('reasoning', ''),
                'alternative_codes': ai_result.get('alternative_codes', [])
            }
            
            db_results.append(db_result)
        
        return db_results
    
    def create_results_dataframe(
        self, 
        original_data: pd.DataFrame, 
        processing_results: BatchProcessingResult
    ) -> pd.DataFrame:
        """
        Создает итоговый DataFrame с результатами
        
        Args:
            original_data: Исходные данные
            processing_results: Результаты обработки AI агентом
            
        Returns:
            DataFrame с результатами
        """
        # Создаем копию исходных данных
        results_df = original_data.copy()
        
        # Добавляем колонки с результатами классификации
        hs_codes = []
        confidences = []
        descriptions = []
        reasonings = []
        alternative_codes = []
        
        for result in processing_results.results:
            hs_codes.append(result.hs_code)
            confidences.append(result.confidence)
            descriptions.append(result.description)
            reasonings.append(result.reasoning)
            alternative_codes.append(", ".join(result.alternative_codes))
        
        # Добавляем новые колонки
        results_df['Код ТН ВЭД'] = hs_codes
        results_df['Уровень доверия'] = confidences
        results_df['Описание ТН ВЭД'] = descriptions
        results_df['Обоснование'] = reasonings
        results_df['Альтернативные коды'] = alternative_codes
        
        # Добавляем колонку с цветовой индикацией
        confidence_status = []
        for conf in confidences:
            if conf >= config.DEFAULT_CONFIDENCE_THRESHOLD:
                confidence_status.append("🟢 Высокий")
            elif conf >= config.MIN_CONFIDENCE_THRESHOLD:
                confidence_status.append("🟡 Средний")
            else:
                confidence_status.append("🔴 Низкий")
        
        results_df['Статус доверия'] = confidence_status
        
        return results_df
    
    def export_results(self, results_df: pd.DataFrame, format: str = "xlsx") -> bytes:
        """
        Экспортирует результаты в указанном формате
        
        Args:
            results_df: DataFrame с результатами
            format: Формат экспорта ('xlsx' или 'csv')
            
        Returns:
            Байты файла для скачивания
        """
        if format.lower() == "xlsx":
            # Экспорт в Excel с форматированием
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                results_df.to_excel(writer, index=False, sheet_name='Результаты классификации')
                
                # Получаем объекты для форматирования
                workbook = writer.book
                worksheet = writer.sheets['Результаты классификации']
                
                # Форматы для разных уровней доверия
                high_confidence_format = workbook.add_format({'bg_color': '#C6EFCE', 'font_color': '#006100'})
                medium_confidence_format = workbook.add_format({'bg_color': '#FFEB9C', 'font_color': '#9C6500'})
                low_confidence_format = workbook.add_format({'bg_color': '#FFC7CE', 'font_color': '#9C0006'})
                
                # Применяем условное форматирование
                confidence_col = results_df.columns.get_loc('Уровень доверия')
                worksheet.conditional_format(
                    1, confidence_col, len(results_df), confidence_col,
                    {'type': 'cell', 'criteria': '>=', 'value': config.DEFAULT_CONFIDENCE_THRESHOLD, 'format': high_confidence_format}
                )
                worksheet.conditional_format(
                    1, confidence_col, len(results_df), confidence_col,
                    {'type': 'cell', 'criteria': 'between', 'minimum': config.MIN_CONFIDENCE_THRESHOLD, 'maximum': config.DEFAULT_CONFIDENCE_THRESHOLD - 1, 'format': medium_confidence_format}
                )
                worksheet.conditional_format(
                    1, confidence_col, len(results_df), confidence_col,
                    {'type': 'cell', 'criteria': '<', 'value': config.MIN_CONFIDENCE_THRESHOLD, 'format': low_confidence_format}
                )
                
                # Автоподгонка ширины колонок
                for idx, col in enumerate(results_df.columns):
                    max_length = max(results_df[col].astype(str).map(len).max(), len(col)) + 2
                    worksheet.set_column(idx, idx, min(max_length, 50))
            
            return output.getvalue()
            
        elif format.lower() == "csv":
            # Экспорт в CSV
            return results_df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
        
        else:
            raise ValueError(f"Неподдерживаемый формат экспорта: {format}")

    def create_dataframe_with_db_results(self, session_id: int) -> pd.DataFrame:
        """
        Создает DataFrame с результатами из базы данных
        
        Args:
            session_id: ID сессии обработки
            
        Returns:
            DataFrame с результатами, включая result_id для связи с БД
        """
        # Получаем результаты из БД
        db_results = self.db_manager.get_session_results(session_id)
        
        if db_results.empty:
            return pd.DataFrame()
        
        # Создаем DataFrame для отображения
        results_df = pd.DataFrame()
        
        # Добавляем скрытую колонку с result_id для связи с БД
        results_df['_result_id'] = db_results['id']
        
        # Основные колонки товара
        results_df['Наименование товара'] = db_results['product_name']
        
        # Дополнительные поля если есть
        if not db_results['original_description'].isna().all():
            results_df['Описание'] = db_results['original_description']
        if not db_results['category'].isna().all():
            results_df['Категория'] = db_results['category']
        if not db_results['brand'].isna().all():
            results_df['Бренд'] = db_results['brand']
        if not db_results['additional_info'].isna().all():
            results_df['Дополнительная информация'] = db_results['additional_info']
        
        # Результаты классификации
        results_df['Код ТН ВЭД'] = db_results['hs_code']
        results_df['Уровень доверия'] = db_results['confidence_percentage']
        results_df['Описание ТН ВЭД'] = db_results['original_description']  # Заглушка, в реальности нужно получать из справочника
        results_df['Обоснование'] = db_results['ai_reasoning']
        
        # Альтернативные коды
        alt_codes = []
        for codes in db_results['alternative_codes']:
            if isinstance(codes, list) and codes:
                alt_codes.append(", ".join(codes))
            else:
                alt_codes.append("")
        results_df['Альтернативные коды'] = alt_codes
        
        # Добавляем колонку с цветовой индикацией
        confidence_status = []
        for conf in db_results['confidence_percentage']:
            if conf >= config.DEFAULT_CONFIDENCE_THRESHOLD:
                confidence_status.append("🟢 Высокий")
            elif conf >= config.MIN_CONFIDENCE_THRESHOLD:
                confidence_status.append("🟡 Средний")
            else:
                confidence_status.append("🔴 Низкий")
        
        results_df['Статус доверия'] = confidence_status
        
        # Пользовательские статусы
        user_statuses = []
        for status in db_results['user_status']:
            if status == 'confirmed':
                user_statuses.append("✅ Подтверждено")
            elif status == 'needs_review':
                user_statuses.append("⚠️ Требует проверки")
            elif status == 'rejected':
                user_statuses.append("❌ Отклонено")
            else:
                user_statuses.append("⏳ Ожидает")
        
        results_df['Пользовательский статус'] = user_statuses
        
        return results_df


class ProgressTracker:
    """Класс для отслеживания прогресса обработки"""
    
    def __init__(self):
        self.progress_bar = None
        self.status_text = None
        self.current_progress = 0
        self.total_items = 0
    
    def initialize(self, total_items: int):
        """Инициализирует трекер прогресса"""
        self.total_items = total_items
        self.current_progress = 0
        
        # Создаем элементы Streamlit для отображения прогресса
        self.status_text = st.empty()
        self.progress_bar = st.progress(0)
        
        self.update_display()
    
    def update(self, processed_items: int, total_items: int = None):
        """Обновляет прогресс"""
        if total_items:
            self.total_items = total_items
        
        self.current_progress = processed_items
        self.update_display()
    
    def update_display(self):
        """Обновляет отображение прогресса"""
        if self.progress_bar and self.status_text:
            progress_ratio = self.current_progress / self.total_items if self.total_items > 0 else 0
            
            self.progress_bar.progress(progress_ratio)
            self.status_text.text(f"Обработано: {self.current_progress} из {self.total_items} позиций ({progress_ratio:.1%})")
    
    def complete(self):
        """Завершает отслеживание прогресса"""
        if self.progress_bar and self.status_text:
            self.progress_bar.progress(1.0)
            self.status_text.text(f"✅ Обработка завершена! Обработано {self.total_items} позиций")
    
    def error(self, error_message: str):
        """Отображает ошибку"""
        if self.status_text:
            self.status_text.error(f"❌ Ошибка: {error_message}")


# Экспортируем основные классы
__all__ = [
    'DataProcessor',
    'ProgressTracker'
] 