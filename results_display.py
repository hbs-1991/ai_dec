"""
AI DECLARANT - Results Display
Модуль для отображения результатов классификации ТН ВЭД
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
from typing import Dict, List, Optional, Any
from datetime import datetime
import config

from ai_agents.hs_agent import BatchProcessingResult, HSCodeSearchResult
from data_processor import DataProcessor
from database_manager import DatabaseManager


def clean_dataframe_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """
    Очищает DataFrame для отображения в интерфейсе
    
    Args:
        df: Исходный DataFrame
        
    Returns:
        Очищенный DataFrame для отображения
    """
    if df.empty:
        return df
    
    display_df = df.copy()
    
    # Скрываем служебные колонки
    columns_to_hide = ['_result_id']
    for col in columns_to_hide:
        if col in display_df.columns:
            # Сохраняем колонку в DataFrame, но не показываем её в списке для отображения
            pass  # Колонка остается в DataFrame для внутреннего использования
    
    # Заменяем NaN на пустые строки для лучшего отображения
    display_df = display_df.fillna('')
    
    # Конвертируем числовые колонки к правильному типу (исправлено deprecated errors='ignore')
    for col in display_df.columns:
        if 'доверия' in col.lower() or 'confidence' in col.lower():
            try:
                # Современный способ без deprecated errors='ignore'
                display_df[col] = pd.to_numeric(display_df[col])
            except (ValueError, TypeError):
                # Если конверсия невозможна, оставляем как есть
                pass
    
    return display_df


class ResultsDisplayManager:
    """
    Менеджер отображения результатов классификации ТН ВЭД кодов
    с поддержкой русского интерфейса и модальных окон
    """
    
    def __init__(self, data_processor, db_manager: DatabaseManager = None):
        self.data_processor = data_processor
        self.db_manager = db_manager or DatabaseManager()
        
        # Русские переводы статусов
        self.status_translations = {
            'pending': 'ожидает',
            'confirmed': 'подтверждено', 
            'needs_review': 'требует проверки',
            'rejected': 'отклонено'
        }
        
        # Цветовая схема статусов
        self.status_colors = {
            'pending': '#6c757d',      # серый
            'confirmed': '#28a745',    # зеленый
            'needs_review': '#ffc107', # желтый
            'rejected': '#dc3545'      # красный
        }
    
    def get_status_badge(self, status: str) -> str:
        """
        Возвращает HTML-бейдж для статуса на русском языке
        
        Args:
            status: Статус ('pending', 'confirmed', 'needs_review', 'rejected')
            
        Returns:
            HTML строка с цветным бейджем
        """
        status_map = {
            'pending': ('ожидает', '#ffc107', '#000'),      # Желтый
            'confirmed': ('подтверждено', '#28a745', '#fff'), # Зеленый  
            'needs_review': ('требует проверки', '#fd7e14', '#fff'), # Оранжевый
            'rejected': ('отклонено', '#dc3545', '#fff')     # Красный
        }
        
        if status in status_map:
            text, bg_color, text_color = status_map[status]
            return f'''
                <span style="
                    background-color: {bg_color}; 
                    color: {text_color}; 
                    padding: 0.25rem 0.5rem; 
                    border-radius: 0.375rem; 
                    font-size: 0.75rem; 
                    font-weight: 600;
                    text-transform: uppercase;
                    letter-spacing: 0.05em;
                    display: inline-block;
                ">
                    {text}
                </span>
            '''
        return f'<span style="color: #6c757d;">{status}</span>'
    
    def get_confidence_badge(self, confidence: int) -> str:
        """Возвращает цветной badge для уровня уверенности"""
        if confidence >= 80:
            color = '#28a745'  # зеленый
            level = 'высокая'
        elif confidence >= 40:
            color = '#ffc107'  # желтый  
            level = 'средняя'
        else:
            color = '#dc3545'  # красный
            level = 'низкая'
        
        return f"""
        <span style="
            background-color: {color}; 
            color: white; 
            padding: 2px 6px; 
            border-radius: 8px; 
            font-size: 11px;
            font-weight: bold;
        ">
            {confidence}% ({level})
        </span>
        """
    
    def render_results_overview(
        self, 
        processing_results: Dict[str, Any],
        results_df: pd.DataFrame
    ):
        """
        Отображает обзор результатов обработки с русскими подписями
        
        Args:
            processing_results: Результаты обработки из data_processor
            results_df: DataFrame с результатами
        """
        st.header("📊 Обзор результатов классификации")
        
        # Получаем статистику из результатов
        stats = processing_results.get("statistics", {})
        processing_time = processing_results.get("processing_time", 0)
        total_items = len(processing_results.get("results", []))
        
        # Основные метрики
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="📦 Всего товаров",
                value=total_items
            )
        
        with col2:
            st.metric(
                label="✅ Высокая уверенность",
                value=f"{stats.get('high', 0)} ({stats.get('high', 0)/total_items*100:.1f}%)" if total_items > 0 else "0 (0%)"
            )
            
        with col3:
            st.metric(
                label="⚠️ Средняя уверенность", 
                value=f"{stats.get('medium', 0)} ({stats.get('medium', 0)/total_items*100:.1f}%)" if total_items > 0 else "0 (0%)"
            )
            
        with col4:
            st.metric(
                label="⏱️ Время обработки",
                value=f"{processing_time:.1f} сек"
            )
        
        st.divider()
        
        # Диаграммы
        col1, col2 = st.columns(2)
        
        with col1:
            # Распределение уверенности (пирог)
            if total_items > 0:
                confidence_data = {
                    'Высокая (≥80%)': stats.get('high', 0),
                    'Средняя (40-79%)': stats.get('medium', 0), 
                    'Низкая (<40%)': stats.get('low', 0)
                }
                
                colors = ['#28a745', '#ffc107', '#dc3545']
                
                fig_pie = px.pie(
                    values=list(confidence_data.values()),
                    names=list(confidence_data.keys()),
                    title="Распределение по уровню уверенности",
                    color_discrete_sequence=colors
                )
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                fig_pie.update_layout(height=400)
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info("Нет данных для отображения диаграммы")
        
        with col2:
            # Гистограмма уверенности
            if not results_df.empty and 'confidence_percentage' in results_df.columns:
                fig_hist = px.histogram(
                    results_df,
                    x='confidence_percentage',
                    nbins=20,
                    title="Гистограмма уровня уверенности",
                    labels={
                        'confidence_percentage': 'Уровень уверенности (%)',
                        'count': 'Количество товаров'
                    }
                )
                fig_hist.update_traces(marker_color='lightblue', marker_line_color='black', marker_line_width=1)
                fig_hist.update_layout(height=400)
                st.plotly_chart(fig_hist, use_container_width=True)
            else:
                st.info("Нет данных для гистограммы")
        
        # Топ HS кодов
        st.subheader("🔝 Наиболее частые коды ТН ВЭД")
        
        if not results_df.empty and 'hs_code' in results_df.columns:
            top_codes = results_df['hs_code'].value_counts().head(10)
            
            if not top_codes.empty:
                fig_bar = px.bar(
                    x=top_codes.values,
                    y=top_codes.index,
                    orientation='h',
                    title="Топ-10 кодов ТН ВЭД",
                    labels={
                        'x': 'Количество товаров',
                        'y': 'Код ТН ВЭД'
                    }
                )
                fig_bar.update_layout(height=400)
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.info("Недостаточно данных для отображения топ кодов")
        else:
            st.info("Нет данных о кодах ТН ВЭД")
        
        # Статистика требующих проверки
        if not results_df.empty and 'confidence_percentage' in results_df.columns:
            needs_review = results_df[results_df['confidence_percentage'] < 80]
            
            if not needs_review.empty:
                st.warning(f"⚠️ **{len(needs_review)} товаров требуют проверки** (уверенность < 80%)")
                
                # Показываем несколько примеров
                with st.expander("🔍 Примеры товаров, требующих проверки", expanded=False):
                    for idx, row in needs_review.head(5).iterrows():
                        product_name = row.get('product_name', 'Не указано')
                        hs_code = row.get('hs_code', 'Не найден')
                        confidence = row.get('confidence_percentage', 0)
                        
                        st.markdown(f"""
                        **{product_name}**
                        - Код ТН ВЭД: `{hs_code}`
                        - Уверенность: {confidence}%
                        """)
            else:
                st.success("✅ Все товары имеют высокую уверенность классификации!")
        

    @st.dialog("🔍 Детальный просмотр товара")
    def render_item_details_modal(self, result_id: int, session_id: int):
        """
        Модальное окно для детального просмотра и редактирования результата
        
        Args:
            result_id: ID результата в базе данных
            session_id: ID сессии обработки
        """
        if not self.db_manager:
            st.error("База данных недоступна")
            return
        
        # Получаем детальную информацию
        result_details = self.db_manager.get_result_details(result_id)
        
        if not result_details:
            st.error("Результат не найден")
            return
        
        # Заголовок
        st.markdown(f"### 📦 {result_details['product_name']}")
        
        # Основная информация в колонках
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"**Файл:** {result_details['filename']}")
            st.markdown(f"**Позиция в файле:** {result_details['row_index'] + 1}")
        
        with col2:
            confidence = result_details['confidence_percentage']
            confidence_color = "🟢" if confidence >= 80 else "🟡" if confidence >= 40 else "🔴"
            st.metric("Уверенность", f"{confidence}%", delta=f"{confidence_color} Уровень")
        
        st.divider()
        
        # Исходные данные товара
        st.subheader("📋 Исходные данные")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if result_details['original_description']:
                st.markdown(f"**Описание:** {result_details['original_description']}")
            if result_details['category']:
                st.markdown(f"**Категория:** {result_details['category']}")
            
        with col2:
            if result_details['brand']:
                st.markdown(f"**Бренд:** {result_details['brand']}")
            if result_details['additional_info']:
                st.markdown(f"**Доп. информация:** {result_details['additional_info']}")
        
        st.divider()
        
        # Результаты классификации
        st.subheader("🎯 Результат классификации AI")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**Код ТН ВЭД:** `{result_details['hs_code']}`")
            
            # Альтернативные коды
            alt_codes = result_details['alternative_codes']
            if alt_codes:
                alt_codes_str = ", ".join(alt_codes)
                st.markdown(f"**Альтернативные коды:** {alt_codes_str}")
        
        with col2:
            # Обоснование в info блоке для лучшей читаемости
            if result_details['ai_reasoning']:
                st.markdown("**Обоснование AI:**")
                st.info(result_details['ai_reasoning'])
        
        st.divider()
        
        # Пользовательские действия
        st.subheader("👤 Действия пользователя")
        
        # Текущий статус
        current_status = result_details['user_status']
        current_notes = result_details['user_notes'] or ""
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("**Статус:**")
            status_options = {
                'confirmed': '✅ Подтверждено',
                'needs_review': '⚠️ Требует проверки', 
                'rejected': '❌ Отклонено'
            }
            
            # Определяем индекс для selectbox
            status_keys = list(status_options.keys())
            current_index = status_keys.index(current_status) if current_status in status_keys else 1
            
            new_status = st.selectbox(
                "Выберите статус:",
                options=status_keys,
                format_func=lambda x: status_options[x],
                index=current_index,
                key=f"status_{result_id}"
            )
        
        with col2:
            st.markdown("**Заметки:**")
            new_notes = st.text_area(
                "Ваши комментарии:",
                value=current_notes,
                height=100,
                placeholder="Добавьте свои заметки о результате классификации...",
                key=f"notes_{result_id}"
            )
        
        # Информация об изменениях
        if result_details['updated_at'] != result_details['created_at']:
            st.caption(f"📝 Последнее обновление: {result_details['updated_at']}")
        
        st.divider()
        
        # Кнопки действий
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if st.button("💾 Сохранить изменения", type="primary", use_container_width=True):
                # Сохраняем изменения в БД
                success = self.db_manager.update_result_user_status(
                    result_id=result_id,
                    user_status=new_status,
                    user_notes=new_notes
                )
                
                if success:
                    st.success("✅ Изменения сохранены!")
                    st.rerun()  # Перезагружаем интерфейс для отображения изменений
                else:
                    st.error("❌ Ошибка сохранения изменений")
        
        with col2:
            if st.button("🔄 Сбросить изменения", use_container_width=True):
                st.rerun()  # Просто перезагружаем модальное окно
        
        with col3:
            if st.button("❌ Закрыть", use_container_width=True):
                st.rerun()  # Закрываем модальное окно

    def render_results_table(self, df: pd.DataFrame, results_per_page: int = 10) -> None:
        """
        Отображает таблицу результатов классификации с пагинацией и фильтрами
        
        Args:
            df: DataFrame с результатами
            results_per_page: Количество результатов на странице
        """
        if df.empty:
            st.warning("📭 Нет результатов для отображения")
            return
        
        # Фильтрация данных
        filtered_df = self._apply_filters(df)
        
        if filtered_df.empty:
            st.warning("🔍 По заданным фильтрам ничего не найдено")
            return
        
        # Отображение количества результатов
        st.markdown(f"**Показано:** {len(filtered_df)} из {len(df)} позиций")
        
        # Настройки таблицы
        with st.expander("⚙️ Настройки таблицы"):
            # Определяем индекс для значения по умолчанию
            options = [5, 10, 25, 50, 100]
            try:
                default_index = options.index(results_per_page)
            except ValueError:
                default_index = 1  # По умолчанию 10 (индекс 1)
            
            results_per_page = st.selectbox(
                "Результатов на странице:",
                options=options,
                index=default_index,
                key="results_per_page"
            )
        
        # Пагинация
        total_pages = (len(filtered_df) - 1) // results_per_page + 1
        
        if total_pages > 1:
            page = st.selectbox(
                f"Страница (1-{total_pages}):",
                range(1, total_pages + 1),
                key="current_page"
            ) - 1
        else:
            page = 0
        
        # Получение данных для текущей страницы
        start_idx = page * results_per_page
        end_idx = start_idx + results_per_page
        page_df = filtered_df.iloc[start_idx:end_idx].copy()
        
        # Отображение данных в виде таблицы с возможностью экспорта
        display_df = clean_dataframe_for_display(page_df)
        st.dataframe(
            display_df,
            use_container_width=True,
            height=400
        )
        
        # 🔍 Детальный просмотр с модальными окнами
        st.markdown("### 🔍 Детальный просмотр")
        st.markdown("Выберите товар для подробного анализа и редактирования статуса:")
        
        # Отображение карточек товаров
        for idx, (_, row) in enumerate(page_df.iterrows()):
            self._render_product_card(row, idx)

    def render_export_section(self, results_df: pd.DataFrame):
        """
        Отображает секцию экспорта результатов
        
        Args:
            results_df: DataFrame с результатами для экспорта
        """
        st.header("💾 Экспорт результатов")
        
        if len(results_df) == 0:
            st.warning("Нет данных для экспорта")
            return
        
        # Очищаем DataFrame перед работой
        clean_df = clean_dataframe_for_display(results_df)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Настройки экспорта
            st.subheader("⚙️ Настройки экспорта")
            
            export_format = st.selectbox(
                "Формат файла",
                options=["Excel (.xlsx)", "CSV (.csv)"],
                help="Выберите формат для экспорта результатов"
            )
            
            # Выбор колонок для экспорта
            export_columns = st.multiselect(
                "Колонки для экспорта",
                options=list(clean_df.columns),
                default=list(clean_df.columns),
                help="Выберите какие колонки включить в экспорт"
            )
            
            # Дополнительные опции
            include_stats = st.checkbox(
                "Включить статистику",
                value=True,
                help="Добавить лист со статистикой (только для Excel)"
            )
        
        with col2:
            st.subheader("📊 Предварительный просмотр")
            
            # Информация о экспорте
            st.metric("Позиций для экспорта", len(clean_df))
            st.metric("Колонок для экспорта", len(export_columns))
            
            # Предварительный просмотр (первые 3 строки)
            if export_columns:
                preview_df = clean_df[export_columns].head(3)
                st.dataframe(preview_df, use_container_width=True)
        
        # Кнопки экспорта
        st.subheader("🚀 Скачать файл")
        
        if not export_columns:
            st.warning("Выберите хотя бы одну колонку для экспорта")
            return
        
        col1, col2 = st.columns(2)
        
        # Подготавливаем данные для экспорта
        export_df = clean_df[export_columns].copy()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        with col1:
            if st.button("📥 Скачать Excel", type="primary", use_container_width=True):
                try:
                    excel_data = self.data_processor.export_results(export_df, "xlsx")
                    filename = f"tnved_results_{timestamp}.xlsx"
                    
                    st.download_button(
                        label="💾 Загрузить Excel файл",
                        data=excel_data,
                        file_name=filename,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
                    st.success(f"✅ Excel файл готов к скачиванию: {filename}")
                    
                except Exception as e:
                    st.error(f"❌ Ошибка создания Excel файла: {str(e)}")
        
        with col2:
            if st.button("📄 Скачать CSV", use_container_width=True):
                try:
                    csv_data = self.data_processor.export_results(export_df, "csv")
                    filename = f"tnved_results_{timestamp}.csv"
                    
                    st.download_button(
                        label="💾 Загрузить CSV файл",
                        data=csv_data,
                        file_name=filename,
                        mime="text/csv",
                        use_container_width=True
                    )
                    st.success(f"✅ CSV файл готов к скачиванию: {filename}")
                    
                except Exception as e:
                    st.error(f"❌ Ошибка создания CSV файла: {str(e)}")
        
        # Информация о форматах
        with st.expander("ℹ️ Информация о форматах экспорта", expanded=False):
            st.markdown("""
            **Excel (.xlsx):**
            - Полное форматирование с цветовым кодированием
            - Условное форматирование по уровню доверия
            - Автоподгонка ширины колонок
            - Возможность включения листа со статистикой
            
            **CSV (.csv):**
            - Простой текстовый формат
            - Совместимость с любыми программами
            - Подходит для дальнейшей обработки данных
            - Кодировка UTF-8 с BOM для корректного отображения
            """)

    def _apply_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Применяет фильтры к DataFrame"""
        
        # Фильтры
        with st.expander("🔧 Фильтры и поиск", expanded=True):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Фильтр по статусу доверия
                if 'Статус доверия' in df.columns:
                    status_options = ["Все"] + list(df['Статус доверия'].unique())
                    selected_status = st.selectbox(
                        "Статус доверия",
                        options=status_options,
                        help="Фильтр по уровню доверия к результату"
                    )
                else:
                    selected_status = "Все"
            
            with col2:
                # Фильтр по диапазону уверенности
                confidence_col = 'Уровень доверия' if 'Уровень доверия' in df.columns else None
                if confidence_col:
                    min_conf, max_conf = st.slider(
                        "Диапазон уверенности (%)",
                        min_value=0,
                        max_value=100,
                        value=(0, 100),
                        help="Фильтр по процентному диапазону уверенности"
                    )
                else:
                    min_conf, max_conf = 0, 100
            
            with col3:
                # Поиск по тексту
                search_text = st.text_input(
                    "Поиск по описанию",
                    placeholder="Введите текст для поиска...",
                    help="Поиск в названиях товаров и описаниях ТН ВЭД"
                )
        
        # Применяем фильтры
        filtered_df = df.copy()
        
        # Фильтр по статусу
        if selected_status != "Все" and 'Статус доверия' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['Статус доверия'] == selected_status]
        
        # Фильтр по диапазону уверенности
        if confidence_col and confidence_col in filtered_df.columns:
            filtered_df = filtered_df[
                (filtered_df[confidence_col] >= min_conf) & 
                (filtered_df[confidence_col] <= max_conf)
            ]
        
        # Поиск по тексту
        if search_text:
            search_columns = []
            for col in filtered_df.columns:
                if not col.startswith('_'):
                    search_columns.append(col)
            
            if search_columns:
                mask = filtered_df[search_columns].astype(str).apply(
                    lambda x: x.str.contains(search_text, case=False, na=False)
                ).any(axis=1)
                filtered_df = filtered_df[mask]
        
        return filtered_df

    def _render_product_card(self, row: pd.Series, idx: int) -> None:
        """Отображает карточку товара с кнопкой подробнее"""
        
        # Получаем основные данные
        product_name = self._get_product_name(row)
        hs_code = self._escape_html(str(row.get('Код ТН ВЭД', '0000.00.000')))
        confidence = row.get('Уровень доверия', 0)
        status = self._get_confidence_status(confidence)
        
        # Компактная карточка товара используя только Streamlit колонки
        # Создаем CSS один раз для всех карточек
        if not hasattr(st.session_state, 'cards_css_loaded'):
            st.markdown("""
            <style>
            .product-card {
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                padding: 12px;
                margin: 4px 0;
                background: white;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }
            </style>
            """, unsafe_allow_html=True)
            st.session_state.cards_css_loaded = True
        
        # Контейнер карточки с CSS классом
        card_container = st.container()
        with card_container:
            # Применяем стиль к контейнеру
            st.markdown('<div class="product-card">', unsafe_allow_html=True)
            
            # Колонки для размещения элементов в одну строку
            col1, col2, col3, col4 = st.columns([4, 2, 3, 1], gap="small")
            
            # Название товара
            with col1:
                st.markdown(f"**📦 {product_name}**")
            
            # Код ТН ВЭД
            with col2:
                st.markdown(f"**Код:** {hs_code}")
            
            # Уверенность со статусом
            with col3:
                # Создаем цветной бейдж для статуса
                badge_html = f"""
                <span style="
                    background: {status['color']};
                    color: white;
                    padding: 2px 8px;
                    border-radius: 12px;
                    font-size: 11px;
                    font-weight: 600;
                    margin-left: 4px;
                ">{status['label']}</span>
                """
                st.markdown(f"**Уверенность:** {confidence}% {badge_html}", unsafe_allow_html=True)
            
            # Кнопка подробнее
            with col4:
                if st.button(
                    "🔍 Подробнее", 
                    key=f"btn_details_{idx}",
                    use_container_width=True,
                    type="secondary",
                    help="Открыть детальную информацию о товаре"
                ):
                    self._show_details_modal(row)
            
            st.markdown('</div>', unsafe_allow_html=True)

    def _get_product_name(self, row: pd.Series) -> str:
        """Получает название товара из строки"""
        # Ищем колонки с названием товара
        possible_columns = ['Наименование товара', 'Product', 'Name', 'Товар']
        
        for col in possible_columns:
            if col in row.index and pd.notna(row[col]):
                # Экранируем HTML символы для безопасности
                return self._escape_html(str(row[col])[:80])
        
        # Если не найдено, берем первую не служебную колонку
        for col in row.index:
            if not col.startswith(('_', 'Код ТН ВЭД', 'Уровень', 'Описание', 'Обоснование', 'Статус')):
                if pd.notna(row[col]):
                    # Экранируем HTML символы для безопасности
                    return self._escape_html(str(row[col])[:80])
        
        return "Товар без названия"
    
    def _escape_html(self, text: str) -> str:
        """Экранирует HTML символы в тексте"""
        import html
        return html.escape(text)

    def _get_confidence_status(self, confidence: float) -> dict:
        """Возвращает статус и цвет для уровня уверенности"""
        if confidence >= 80:
            return {"label": "Высокий", "color": "#10b981"}
        elif confidence >= 40:
            return {"label": "Средний", "color": "#f59e0b"}
        else:
            return {"label": "Низкий", "color": "#ef4444"}

    @st.dialog("🔍 Детальная информация о товаре", width="large")
    def _show_details_modal(self, row: pd.Series) -> None:
        """Показывает модальное окно с детальной информацией"""
        
        # Получаем данные
        product_name = self._get_product_name(row)
        hs_code = self._escape_html(str(row.get('Код ТН ВЭД', 'Не определен')))
        confidence = row.get('Уровень доверия', 0)
        description = self._escape_html(str(row.get('Описание ТН ВЭД', 'Описание недоступно')))
        reasoning = self._escape_html(str(row.get('Обоснование', 'Обоснование недоступно')))
        status = self._get_confidence_status(confidence)
        
        # Стильный заголовок
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 16px;
            border-radius: 8px;
            margin-bottom: 15px;
            text-align: center;
        ">
            <h2 style="margin: 0; font-size: 20px; font-weight: 700;">
                📦 {product_name}
            </h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Основная информация в улучшенных контейнерах
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div style="
                background: #f8fafc;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                padding: 12px;
                margin-bottom: 10px;
            ">
                <h4 style="color: #1e293b; margin: 0; font-size: 16px;">🎯 Результат классификации</h4>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div style="padding: 0 12px; font-size: 14px;">
                <p style="margin: 8px 0;"><strong style="color: #475569;">Код ТН ВЭД:</strong></p>
                <code style="
                    background: #f1f5f9; 
                    padding: 8px 12px; 
                    border-radius: 6px; 
                    color: #0f172a;
                    font-size: 16px;
                    font-weight: 700;
                    border: 2px solid #e2e8f0;
                    display: block;
                    text-align: center;
                    margin-bottom: 12px;
                ">{hs_code}</code>
                <p style="margin: 8px 0;"><strong style="color: #475569;">Уровень доверия:</strong> 
                   <span style="color: {status['color']}; font-weight: 700; font-size: 16px;">
                       {confidence}% ({status['label']})
                   </span>
                </p>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown("""
            <div style="
                background: #f8fafc;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                padding: 12px;
                margin-bottom: 10px;
            ">
                <h4 style="color: #1e293b; margin: 0; font-size: 16px;">📊 Статус уверенности</h4>
            </div>
            """, unsafe_allow_html=True)
            
            confidence_color = status['color']
            st.markdown(f"""
            <div style="padding: 0 12px;">
                <div style="
                    background: linear-gradient(135deg, {confidence_color}20 0%, {confidence_color}10 100%);
                    border: 2px solid {confidence_color};
                    border-radius: 8px;
                    padding: 12px;
                    text-align: center;
                    margin-bottom: 8px;
                ">
                    <div style="
                        color: {confidence_color}; 
                        font-weight: 700; 
                        font-size: 14px; 
                        margin-bottom: 6px;
                    ">
                        {status['label'].upper()} УРОВЕНЬ
                    </div>
                    <div style="
                        color: {confidence_color}; 
                        font-size: 24px; 
                        font-weight: 800;
                    ">
                        {confidence}%
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Разделитель
        st.markdown("---")
        
        # Описание ТН ВЭД в улучшенном контейнере
        st.markdown("""
        <div style="
            background: #fef7ed;
            border: 1px solid #fed7aa;
            border-radius: 8px;
            padding: 12px;
            margin: 15px 0 10px 0;
        ">
            <h4 style="color: #9a3412; margin: 0; font-size: 16px;">📋 Описание по ТН ВЭД</h4>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div style="
            background: white;
            border: 1px solid #e5e7eb;
            border-radius: 6px;
            padding: 12px;
            font-size: 14px;
            line-height: 1.5;
            color: #374151;
            margin-bottom: 10px;
        ">
            {description}
        </div>
        """, unsafe_allow_html=True)
        
        # Обоснование в улучшенном контейнере
        st.markdown("""
        <div style="
            background: #f0f9ff;
            border: 1px solid #bae6fd;
            border-radius: 8px;
            padding: 12px;
            margin: 15px 0 10px 0;
        ">
            <h4 style="color: #0c4a6e; margin: 0; font-size: 16px;">💭 Обоснование выбора</h4>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div style="
            background: white;
            border: 1px solid #e5e7eb;
            border-radius: 6px;
            padding: 12px;
            font-size: 14px;
            line-height: 1.5;
            color: #374151;
            margin-bottom: 10px;
        ">
            {reasoning}
        </div>
        """, unsafe_allow_html=True)
        
        # Исходные данные товара
        st.markdown("""
        <div style="
            background: #f0fdf4;
            border: 1px solid #bbf7d0;
            border-radius: 8px;
            padding: 12px;
            margin: 15px 0 10px 0;
        ">
            <h4 style="color: #14532d; margin: 0; font-size: 16px;">📋 Исходные данные товара</h4>
        </div>
        """, unsafe_allow_html=True)
        
        # Создаем таблицу с исходными данными
        original_data = []
        for col_name, value in row.items():
            if not col_name.startswith(('_', 'Код ТН ВЭД', 'Уровень', 'Описание ТН ВЭД', 'Обоснование', 'Статус')):
                if pd.notna(value) and str(value).strip():
                    original_data.append({'Поле': col_name, 'Значение': str(value)})
        
        if original_data:
            original_df = pd.DataFrame(original_data)
            st.dataframe(
                original_df, 
                use_container_width=True, 
                hide_index=True,
                height=min(len(original_data) * 35 + 50, 300)
            )
        else:
            st.warning("⚠️ Нет дополнительных данных о товаре")
        
        # Улучшенная кнопка закрытия
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button(
                "✅ Закрыть окно", 
                type="primary", 
                use_container_width=True,
                key="close_modal_btn"
            ):
                st.rerun()


# Экспортируем основные классы
__all__ = [
    'ResultsDisplayManager'
] 