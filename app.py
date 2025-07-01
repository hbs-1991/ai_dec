"""
AI DECLARANT MVP - Главное приложение
Автоматическое определение кодов ТН ВЭД для таможенных брокеров Туркменистана
"""

import streamlit as st
import pandas as pd
import io
import asyncio
from typing import Optional, Dict, Any
import config
from datetime import datetime
import os
import time

from data_processor import DataProcessor, ProgressTracker
from results_display import ResultsDisplayManager
from database_manager import DatabaseManager

# Настройка страницы
st.set_page_config(
    page_title="AI DECLARANT - Определение кодов ТН ВЭД",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS стили для улучшения дизайна
st.markdown("""
<style>
    /* Улучшенная типографика */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Заголовки */
    h1 {
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        color: #1f2937 !important;
        margin-bottom: 1rem !important;
    }
    
    h2 {
        font-size: 1.8rem !important;
        font-weight: 600 !important;
        color: #374151 !important;
        margin-top: 2rem !important;
        margin-bottom: 1rem !important;
    }
    
    h3 {
        font-size: 1.4rem !important;
        font-weight: 600 !important;
        color: #4b5563 !important;
        margin-top: 1.5rem !important;
        margin-bottom: 0.8rem !important;
    }
    
    /* Улучшенные кнопки */
    .stButton > button {
        font-size: 1rem !important;
        font-weight: 500 !important;
        padding: 0.5rem 1.5rem !important;
        border-radius: 8px !important;
        border: none !important;
        transition: all 0.2s ease !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15) !important;
    }
    
    /* Улучшенные карточки метрик */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 0.5rem 0;
    }
    
    .metric-title {
        font-size: 0.9rem;
        font-weight: 500;
        opacity: 0.9;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
    }
    
    /* Прогресс бар */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Боковая панель */
    .css-1d391kg {
        background: #f8fafc;
    }
    
    /* Улучшенные алерты */
    .stAlert {
        border-radius: 8px !important;
        border: none !important;
        font-size: 1rem !important;
    }
    
    /* Файл аплоадер */
    .stFileUploader > div {
        border-radius: 12px !important;
        border: 2px dashed #d1d5db !important;
        padding: 2rem !important;
    }
    
    /* Таблицы */
    .stDataFrame {
        border-radius: 8px !important;
        overflow: hidden !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1) !important;
    }
    
    /* Селектбоксы */
    .stSelectbox > div > div {
        border-radius: 8px !important;
        border: 1px solid #d1d5db !important;
    }
    
    /* Текстовые поля */
    .stTextInput > div > div {
        border-radius: 8px !important;
        border: 1px solid #d1d5db !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        font-size: 1.1rem !important;
        font-weight: 600 !important;
    }
    
    /* Код блоки */
    code {
        background: #f3f4f6 !important;
        padding: 0.2rem 0.4rem !important;
        border-radius: 4px !important;
        font-size: 0.9rem !important;
        color: #374151 !important;
    }
</style>
""", unsafe_allow_html=True)

# Инициализация session state
def init_session_state():
    """Инициализация состояния сессии"""
    if "uploaded_file" not in st.session_state:
        st.session_state.uploaded_file = None
    if "uploaded_data" not in st.session_state:
        st.session_state.uploaded_data = None
    if "column_mapping" not in st.session_state:
        st.session_state.column_mapping = {}
    if "processing_results" not in st.session_state:
        st.session_state.processing_results = None
    if "data_processor" not in st.session_state:
        st.session_state.data_processor = DataProcessor()
    if "current_step" not in st.session_state:
        st.session_state.current_step = "upload"

def validate_file(uploaded_file) -> tuple[bool, str]:
    """Валидация загруженного файла"""
    if uploaded_file is None:
        return False, "Файл не выбран"
    
    # Проверка типа файла
    file_extension = uploaded_file.name.split('.')[-1].lower()
    if file_extension not in config.SUPPORTED_FILE_TYPES:
        return False, f"Неподдерживаемый тип файла. Поддерживаются: {', '.join(config.SUPPORTED_FILE_TYPES)}"
    
    # Проверка размера файла
    if uploaded_file.size > config.MAX_FILE_SIZE_MB * 1024 * 1024:
        return False, f"Файл слишком большой. Максимальный размер: {config.MAX_FILE_SIZE_MB}MB"
    
    return True, "OK"

def load_file_data(uploaded_file) -> Optional[pd.DataFrame]:
    """Загрузка данных из файла"""
    try:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        if file_extension == 'csv':
            # Пробуем разные кодировки для CSV
            for encoding in ['utf-8', 'windows-1251', 'cp1251']:
                try:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                st.error("Не удалось определить кодировку CSV файла")
                return None
        
        elif file_extension in ['xlsx', 'xls']:
            df = pd.read_excel(uploaded_file)
        
        else:
            st.error(f"Неподдерживаемый формат файла: {file_extension}")
            return None
        
        # Проверка количества строк
        if len(df) > config.MAX_ROWS_PER_FILE:
            st.warning(f"Файл содержит {len(df)} строк. Будут обработаны только первые {config.MAX_ROWS_PER_FILE} строк.")
            df = df.head(config.MAX_ROWS_PER_FILE)
        
        # Удаляем пустые строки
        df = df.dropna(how='all')
        
        return df
    
    except Exception as e:
        st.error(f"Ошибка при загрузке файла: {str(e)}")
        return None

def render_file_upload():
    """Отрисовка интерфейса загрузки файла"""
    st.header("📁 Загрузка файла")
    
    # Информация о требованиях к файлу
    with st.expander("ℹ️ Требования к файлу", expanded=False):
        st.markdown(f"""
        **Поддерживаемые форматы:** {', '.join(config.SUPPORTED_FILE_TYPES).upper()}  
        **Максимальный размер:** {config.MAX_FILE_SIZE_MB}MB  
        **Максимальное количество строк:** {config.MAX_ROWS_PER_FILE:,}  
        
        **Рекомендуемая структура:**
        - Первая строка - заголовки колонок
        - Одна из колонок должна содержать название/описание товара
        - Дополнительные колонки с описанием, категорией, брендом (опционально)
        """)
    
    # Загрузка файла
    uploaded_file = st.file_uploader(
        "Выберите файл с товарами",
        type=config.SUPPORTED_FILE_TYPES,
        help="Перетащите файл сюда или нажмите для выбора"
    )
    
    if uploaded_file is not None:
        # Валидация файла
        is_valid, error_message = validate_file(uploaded_file)
        
        if is_valid:
            st.success(f"✅ Файл загружен: {uploaded_file.name}")
            
            # Загрузка данных
            with st.spinner("Загружаем данные..."):
                df = load_file_data(uploaded_file)
            
            if df is not None:
                st.session_state.uploaded_file = uploaded_file
                st.session_state.uploaded_data = df
                st.session_state.current_step = "preview"
                st.rerun()
        else:
            st.error(f"❌ {error_message}")

def render_data_preview():
    """Отрисовка предпросмотра данных"""
    if st.session_state.uploaded_data is None:
        st.error("Данные не загружены")
        return
    
    df = st.session_state.uploaded_data
    
    st.header("👀 Предпросмотр данных")
    
    # Информация о файле
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Строк", len(df))
    with col2:
        st.metric("Колонок", len(df.columns))
    with col3:
        st.metric("Размер", f"{st.session_state.uploaded_file.size / 1024:.1f} KB")
    with col4:
        st.metric("Формат", st.session_state.uploaded_file.name.split('.')[-1].upper())
    
    # Отображение первых строк
    st.subheader("📋 Первые 10 строк")
    st.dataframe(df.head(10), use_container_width=True)
    
    # Информация о колонках
    st.subheader("📊 Информация о колонках")
    column_info = pd.DataFrame({
        'Колонка': df.columns,
        'Тип данных': [str(dtype) for dtype in df.dtypes],  # Преобразуем в строки
        'Заполненность': [f"{(~df[col].isna()).sum()}/{len(df)} ({(~df[col].isna()).sum()/len(df)*100:.1f}%)" for col in df.columns],
        'Примеры значений': [str(df[col].dropna().iloc[0] if len(df[col].dropna()) > 0 else 'Нет данных')[:50] + '...' if len(str(df[col].dropna().iloc[0] if len(df[col].dropna()) > 0 else 'Нет данных')) > 50 else str(df[col].dropna().iloc[0] if len(df[col].dropna()) > 0 else 'Нет данных') for col in df.columns]
    })
    st.dataframe(column_info, use_container_width=True, hide_index=True)
    
    # Кнопки навигации
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("⬅️ Назад к загрузке", use_container_width=True):
            st.session_state.current_step = "upload"
            st.rerun()
    
    with col2:
        if st.button("➡️ Сопоставить колонки", use_container_width=True, type="primary"):
            st.session_state.current_step = "mapping"
            st.rerun()

def render_column_mapping():
    """Отрисовка интерфейса сопоставления колонок"""
    if st.session_state.uploaded_data is None:
        st.error("Данные не загружены")
        return
    
    df = st.session_state.uploaded_data
    
    st.header("🔗 Сопоставление колонок")
    st.markdown("Укажите, какие колонки содержат информацию о товарах:")
    
    # Форма сопоставления
    with st.form("column_mapping_form"):
        mapping = {}
        
        # Обязательная колонка - название товара
        st.subheader("📝 Обязательные поля")
        mapping['product_name'] = st.selectbox(
            "Наименование товара *",
            options=[None] + list(df.columns),
            help="Выберите колонку с названиями товаров"
        )
        
        # Опциональные колонки
        st.subheader("📋 Дополнительные поля (опционально)")
        col1, col2 = st.columns(2)
        
        with col1:
            mapping['description'] = st.selectbox(
                "Описание товара",
                options=[None] + list(df.columns),
                help="Колонка с подробным описанием товара"
            )
            
            mapping['category'] = st.selectbox(
                "Категория",
                options=[None] + list(df.columns),
                help="Колонка с категорией товара"
            )
        
        with col2:
            mapping['brand'] = st.selectbox(
                "Бренд/Производитель",
                options=[None] + list(df.columns),
                help="Колонка с информацией о бренде или производителе"
            )
            
            mapping['additional_info'] = st.selectbox(
                "Дополнительная информация",
                options=[None] + list(df.columns),
                help="Любая другая полезная информация о товаре"
            )
        
        # Проверка сопоставления
        if mapping['product_name'] is None:
            st.error("⚠️ Необходимо выбрать колонку с наименованием товара!")
        else:
            # Показываем примеры сопоставления
            st.subheader("🔍 Предварительный просмотр сопоставления")
            
            preview_data = []
            for i in range(min(3, len(df))):
                row_data = {}
                for field, column in mapping.items():
                    if column is not None:
                        value = str(df.iloc[i][column])[:100]
                        if len(str(df.iloc[i][column])) > 100:
                            value += "..."
                        row_data[config.REQUIRED_COLUMNS.get(field, field)] = value
                    else:
                        row_data[config.REQUIRED_COLUMNS.get(field, field)] = "—"
                preview_data.append(row_data)
            
            preview_df = pd.DataFrame(preview_data)
            st.dataframe(preview_df, use_container_width=True)
        
        # Кнопки отправки
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if st.form_submit_button("⬅️ Назад", use_container_width=True):
                st.session_state.current_step = "preview"
                st.rerun()
        
        with col3:
            if st.form_submit_button("🚀 Начать обработку", use_container_width=True, type="primary"):
                if mapping['product_name'] is not None:
                    st.session_state.column_mapping = mapping
                    st.session_state.current_step = "processing"
                    st.rerun()
                else:
                    st.error("Необходимо выбрать колонку с наименованием товара!")

def render_processing():
    """Этап обработки данных с помощью AI агента"""
    if st.session_state.uploaded_data is None or not st.session_state.column_mapping:
        st.error("Данные не готовы для обработки")
        return
    
    st.header("🤖 Обработка данных с помощью AI")
    
    df = st.session_state.uploaded_data
    mapping = st.session_state.column_mapping
    
    # Подготовка элементов для обработки
    items = st.session_state.data_processor.prepare_items_for_processing(df, mapping)
    
    # Показываем информацию о предстоящей обработке
    st.subheader("📊 Информация об обработке")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Позиций к обработке", len(items))
    with col2:
        estimated_time = len(items) * 2  # Примерно 2 секунды на позицию
        st.metric("Примерное время", f"{estimated_time//60}:{estimated_time%60:02d}")
    with col3:
        st.metric("Размер пакета", config.BATCH_SIZE)
    
    # Предварительный просмотр первых элементов
    with st.expander("🔍 Предварительный просмотр данных для AI", expanded=False):
        for i, item in enumerate(items[:3]):
            st.markdown(f"**Позиция {i+1}:**")
            st.json(item)
    
    # Кнопка запуска обработки
    if st.button("🚀 Начать обработку с помощью AI", type="primary", use_container_width=True):
        
        # Проверяем наличие API ключа
        if not config.OPENAI_API_KEY:
            st.error("❌ Не настроен OPENAI_API_KEY. Проверьте файл .env")
            return
        
        # Запускаем обработку
        process_with_ai(items)
    
    # Навигация
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("⬅️ Назад к сопоставлению", use_container_width=True):
            st.session_state.current_step = "mapping"
            st.rerun()
    
    with col2:
        if st.session_state.processing_results is not None:
            if st.button("➡️ Посмотреть результаты", use_container_width=True):
                st.session_state.current_step = "results"
                st.rerun()

def process_with_ai(items: list):
    """Асинхронная обработка данных с помощью AI агента"""
    
    # Инициализируем трекер прогресса
    progress_tracker = ProgressTracker()
    progress_tracker.initialize(len(items))
    
    # Функция обновления прогресса
    def update_progress(processed: int, total: int):
        progress_tracker.update(processed, total)
    
    try:
        # Запускаем асинхронную обработку
        with st.spinner("Анализ товаров с помощью AI..."):
            # Создаем новый event loop для Streamlit
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                # Получаем имя файла для истории
                filename = st.session_state.uploaded_file.name if st.session_state.uploaded_file else "unknown_file"
                
                # Выполняем обработку с новой сигнатурой
                result = loop.run_until_complete(
                    st.session_state.data_processor.process_data_with_ai(
                        data=st.session_state.uploaded_data,
                        column_mapping=st.session_state.column_mapping,
                        filename=filename,
                        progress_callback=update_progress
                    )
                )
                
                # Проверяем успешность обработки
                if not result["success"]:
                    st.error(f"❌ Ошибка обработки: {result.get('error', 'Неизвестная ошибка')}")
                    return
                
                # Сохраняем результаты в session state
                st.session_state.processing_results = result
                st.session_state.results_dataframe = result["dataframe"]
                st.session_state.current_session_id = result["session_id"]
                
                # Показываем статистику
                progress_tracker.complete()
                
                # Отображаем итоговую статистику
                st.success("🎉 Обработка завершена и сохранена в базу данных!")
                
                stats = result["statistics"]
                processing_time = result["processing_time"]
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Обработано", len(result["results"]))
                with col2:
                    st.metric("Высокое доверие", f"{stats['high']} (≥80%)")
                with col3:
                    st.metric("Среднее доверие", f"{stats['medium']} (40-79%)")
                with col4:
                    st.metric("Время обработки", f"{processing_time:.1f}с")
                
                # Автоматически переходим к результатам
                st.session_state.current_step = "results"
                st.rerun()
                
            finally:
                loop.close()
                
    except Exception as e:
        progress_tracker.error(str(e))
        st.error(f"Критическая ошибка обработки: {str(e)}")
        
        # Сохраняем ошибку в состояние для анализа
        st.session_state.processing_error = str(e)


def render_sidebar():
    """Отображает боковую панель с информацией и метриками"""
    with st.sidebar:
        # Логотип и заголовок
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0; border-bottom: 1px solid #e5e7eb; margin-bottom: 1.5rem;">
            <h2 style="margin: 0; color: #1f2937; font-size: 1.5rem;">📊 Статистика</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Статистики файла
        if hasattr(st.session_state, 'uploaded_data') and st.session_state.uploaded_data is not None:
            df = st.session_state.uploaded_data
            
            # Создаем красивые метрики
            metrics = [
                {"title": "Загруженных строк", "value": str(len(df)), "icon": "📦"},
                {"title": "Колонок", "value": str(len(df.columns)), "icon": "📋"},
            ]
            
            # Добавляем метрику сопоставленных полей
            if hasattr(st.session_state, 'column_mapping') and st.session_state.column_mapping:
                mapped_count = sum(1 for v in st.session_state.column_mapping.values() if v is not None)
                metrics.append({"title": "Сопоставленных полей", "value": str(mapped_count), "icon": "🔗"})
            
            # Отображаем метрики
            for metric in metrics:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">{metric['icon']} {metric['title']}</div>
                    <div class="metric-value">{metric['value']}</div>
                </div>
                """, unsafe_allow_html=True)
        
        # Разделитель
        st.markdown("<div style='margin: 2rem 0;'></div>", unsafe_allow_html=True)
        
        # Справочная информация
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0; border-bottom: 1px solid #e5e7eb; margin-bottom: 1.5rem;">
            <h2 style="margin: 0; color: #1f2937; font-size: 1.5rem;">ℹ️ Справка</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Информационные блоки
        info_sections = [
            {
                "title": "Поддерживаемые форматы:",
                "items": ["📄 Excel (.xlsx, .xls)", "📄 CSV (.csv)"],
                "color": "#3b82f6"
            },
            {
                "title": "Ограничения:",
                "items": ["📏 Размер файла: до 50MB", "📊 Количество строк: до 1,000"],
                "color": "#f59e0b"
            },
            {
                "title": "Требования:",
                "items": ["🏷️ Файл должен содержать заголовки", "📝 Обязательно наличие колонки с названием товара"],
                "color": "#10b981"
            }
        ]
        
        for section in info_sections:
            st.markdown(f"""
            <div style="
                background: {section['color']}15;
                border-left: 4px solid {section['color']};
                padding: 1rem;
                margin: 1rem 0;
                border-radius: 0 8px 8px 0;
            ">
                <div style="font-weight: 600; color: {section['color']}; margin-bottom: 0.5rem;">
                    {section['title']}
                </div>
                <div style="font-size: 0.9rem; color: #4b5563;">
                    {"<br>".join(section['items'])}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Системная информация
        st.markdown("<div style='margin: 2rem 0;'></div>", unsafe_allow_html=True)
        
        # Информация о версии и состоянии
        st.markdown("""
        <div style="
            background: #f3f4f6;
            padding: 1rem;
            border-radius: 8px;
            text-align: center;
            font-size: 0.8rem;
            color: #6b7280;
            margin-top: 2rem;
        ">
            <div style="font-weight: 600; margin-bottom: 0.5rem;">AI DECLARANT v1.0</div>
            <div>Powered by OpenAI & Streamlit</div>
        </div>
        """, unsafe_allow_html=True)

def main():
    """Главная функция приложения"""
    # Инициализация
    init_session_state()
    
    # Заголовок приложения
    st.title(f"{config.APP_ICON} {config.APP_TITLE}")
    st.markdown(config.APP_DESCRIPTION)
    
    # Прогресс по шагам
    steps = ["Загрузка файла", "Предпросмотр", "Сопоставление", "Обработка", "Результаты"]
    current_step_index = {
        "upload": 0,
        "preview": 1, 
        "mapping": 2,
        "processing": 3,
        "results": 4
    }.get(st.session_state.current_step, 0)
    
    # Показываем прогресс
    st.progress((current_step_index + 1) / len(steps))
    st.markdown(f"**Шаг {current_step_index + 1} из {len(steps)}: {steps[current_step_index]}**")
    
    st.divider()
    
    # Боковая панель с информацией
    render_sidebar()
    
    # Основной контент в зависимости от текущего шага
    if st.session_state.current_step == "upload":
        render_file_upload()
    elif st.session_state.current_step == "preview":
        render_data_preview()
    elif st.session_state.current_step == "mapping":
        render_column_mapping()
    elif st.session_state.current_step == "processing":
        render_processing()
    elif st.session_state.current_step == "results":
        # Шаг 5: Результаты
        if st.session_state.processing_results is not None:
            # Создаем объекты для отображения
            results_display = ResultsDisplayManager(
                data_processor=st.session_state.data_processor,
                db_manager=DatabaseManager() if 'current_session_id' in st.session_state else None
            )
            
            # Используем табы для организации результатов
            tab1, tab2, tab3 = st.tabs(["📊 Обзор", "📋 Результаты", "💾 Экспорт"])
            
            # Создаем DataFrame с результатами
            data_processor = st.session_state.data_processor
            if hasattr(st.session_state, 'current_session_id') and st.session_state.current_session_id:
                results_df = data_processor.create_dataframe_with_db_results(st.session_state.current_session_id)
            else:
                # Fallback на создание DataFrame из processing_results
                results_df = data_processor.create_results_dataframe(st.session_state.processing_results)
                
            with tab1:
                results_display.render_results_overview(st.session_state.processing_results, results_df)
            
            with tab2:
                # Используем новый улучшенный интерфейс
                results_display.render_results_table(results_df)
            
            with tab3:
                results_display.render_export_section(results_df)
        else:
            st.error("Результаты не найдены. Пожалуйста, выполните обработку данных.")
        
        # Навигационные кнопки
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("⬅️ Вернуться к сопоставлению", type="secondary"):
                st.session_state.current_step = "mapping"
                st.rerun()
        
        with col2:
            if st.button("🔄 Повторить обработку", type="secondary"):
                st.session_state.current_step = "processing"
                st.rerun()
        
        with col3:
            if st.button("📁 Загрузить новый файл", type="primary"):
                # Сброс состояния для новой загрузки
                for key in ['uploaded_file', 'uploaded_data', 'column_mapping', 'processing_results']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.session_state.current_step = "upload"
                st.rerun()
    else:
        st.error("Неизвестный шаг")

if __name__ == "__main__":
    main() 