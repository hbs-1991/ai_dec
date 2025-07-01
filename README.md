# AI DECLARANT 🤖

**Автоматическое определение кодов ТН ВЭД для таможенных брокеров Туркменистана**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4-green.svg)](https://openai.com)

## 📋 Описание

AI DECLARANT - это веб-приложение для автоматической классификации товаров по кодам ТН ВЭД (Товарная номенклатура внешнеэкономической деятельности) с использованием искусственного интеллекта. Разработано специально для таможенных брокеров и декларантов Туркменистана.

### ✨ Основные возможности

- 📊 **Пакетная обработка**: Загрузка и обработка Excel/CSV файлов до 1000 позиций
- 🤖 **AI-классификация**: Автоматическое определение кодов ТН ВЭД с помощью OpenAI GPT-4
- 📈 **Аналитика результатов**: Детальная статистика с уровнем доверия
- 💾 **База данных**: Сохранение истории обработок в SQLite
- 📁 **Экспорт данных**: Выгрузка результатов в Excel/CSV
- 🔍 **Детальный просмотр**: Модальные окна с обоснованием AI
- 🎨 **Современный UI**: Красивый интерфейс на Streamlit

## 🚀 Быстрый старт

### Требования

- Python 3.9+
- OpenAI API ключ
- Доступ к интернету

### Установка

1. **Клонирование репозитория**
```bash
git clone https://github.com/hbs-1991/ai_dec.git
cd ai_dec
```

2. **Установка зависимостей**
```bash
pip install -r requirements.txt
```

3. **Настройка окружения**
```bash
# Создайте файл .env на основе env_example.txt
cp env_example.txt .env
```

4. **Настройка OpenAI API**
Добавьте ваш OpenAI API ключ в файл `.env`:
```
OPENAI_API_KEY=your-api-key-here
```

5. **Запуск приложения**
```bash
streamlit run app.py
```

Приложение будет доступно по адресу: `http://localhost:8501`

## 📖 Использование

### 1. Загрузка данных
- Поддерживаемые форматы: Excel (.xlsx) и CSV
- Автоматическое сопоставление колонок
- Предпросмотр загруженных данных

### 2. Обработка
- Выбор AI агента для классификации
- Асинхронная обработка товаров
- Отслеживание прогресса в реальном времени

### 3. Анализ результатов
- Статистика по уровням доверия
- Цветовое кодирование результатов
- Фильтрация и поиск

### 4. Экспорт
- Выгрузка в Excel с форматированием
- Настраиваемые колонки экспорта
- Сохранение пользовательских заметок

## 🏗️ Архитектура

```
ai_dec/
├── ai_agents/              # AI агенты
│   ├── __init__.py
│   ├── hs_agent.py        # Агент поиска кодов ТН ВЭД
│   └── code_search_agent.py
├── app.py                 # Главное приложение Streamlit
├── data_processor.py      # Обработка данных
├── database_manager.py    # Работа с БД
├── results_display.py     # Отображение результатов
├── config.py             # Конфигурация
├── requirements.txt      # Зависимости
├── test_data/           # Тестовые данные
└── README.md           # Документация
```

## 🛠️ Технологии

- **Backend**: Python 3.9+
- **Frontend**: Streamlit 1.28+
- **AI**: OpenAI GPT-4, openai-agents-python
- **Database**: SQLite
- **Data Processing**: pandas, openpyxl
- **UI Components**: st.dialog, st.columns, st.container

## 📊 Особенности AI агента

### HSCodeSearchAgent
- Поиск в базе знаний OpenAI Vector Store
- Веб-поиск для актуализации данных
- Многоступенчатый анализ товаров
- Обоснование выбора кода ТН ВЭД

### Уровни доверия
- 🟢 **Высокий** (80-100%): Надежная классификация
- 🟡 **Средний** (40-79%): Требует проверки
- 🔴 **Низкий** (0-39%): Нуждается в ручной корректировке

## 🎨 UI/UX особенности

- **Компактные карточки товаров** в одну строку
- **Модальные окна** для детального просмотра
- **Цветовое кодирование** по уровню доверия
- **Адаптивный дизайн** для разных экранов
- **Современная типографика** и иконки

## 🔧 Конфигурация

### config.py
```python
# Настройки AI агента
MAX_BATCH_SIZE = 1000
CONFIDENCE_THRESHOLD = 40
DEFAULT_AGENT_TYPE = "hs_agent"

# Настройки UI
RESULTS_PER_PAGE = 10
SHOW_DEBUG_INFO = False
```

## 📈 Мониторинг и логирование

- Сохранение всех сессий обработки
- История изменений пользователя
- Метрики производительности AI
- Экспорт аудита операций

## 🚀 Развертывание

### Streamlit Cloud
```bash
# Подготовка для Streamlit Cloud
echo "OPENAI_API_KEY = 'your-key'" > .streamlit/secrets.toml
```

### Docker (опционально)
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
```

## 🤝 Участие в разработке

1. Fork репозитория
2. Создайте ветку функции (`git checkout -b feature/amazing-feature`)
3. Commit изменения (`git commit -m 'Add amazing feature'`)
4. Push в ветку (`git push origin feature/amazing-feature`)
5. Откройте Pull Request

## 📝 Changelog

### v1.0.0 (Текущая версия)
- ✅ Базовая функциональность классификации
- ✅ Веб-интерфейс на Streamlit
- ✅ Интеграция с OpenAI GPT-4
- ✅ База данных SQLite
- ✅ Экспорт результатов
- ✅ Модальные окна для детального просмотра

### Планируемые функции
- 🔄 API для интеграции
- 🔄 Пользовательские роли
- 🔄 Расширенная аналитика
- 🔄 Интеграция с таможенными системами

## 📞 Поддержка

- 📧 Email: [создать issue](https://github.com/hbs-1991/ai_dec/issues)
- 📋 Документация: [wiki](https://github.com/hbs-1991/ai_dec/wiki)
- 🐛 Баги: [issues](https://github.com/hbs-1991/ai_dec/issues)

## 📄 Лицензия

Этот проект лицензирован под MIT License - см. файл [LICENSE](LICENSE) для деталей.

## 🙏 Благодарности

- OpenAI за GPT-4 API
- Streamlit за отличный фреймворк
- Сообщество разработчиков Python

---

**AI DECLARANT** - делаем таможенную классификацию проще и точнее! 🎯 