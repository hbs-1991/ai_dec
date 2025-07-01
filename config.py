"""
Конфигурация AI DECLARANT MVP
"""
import os
from dotenv import load_dotenv

# Загружаем переменные окружения
load_dotenv()

# Настройки приложения
APP_TITLE = "AI DECLARANT - Определение кодов ТН ВЭД"
APP_ICON = "📋"
APP_DESCRIPTION = "Автоматическое определение кодов ТН ВЭД для таможенных брокеров Туркменистана"

# Ограничения файлов
MAX_FILE_SIZE_MB = 50
MAX_ROWS_PER_FILE = 1000
SUPPORTED_FILE_TYPES = ["xlsx", "xls", "csv"]

# Настройки обработки
DEFAULT_CONFIDENCE_THRESHOLD = 80  # Высокий уровень доверия (≥80%)
MIN_CONFIDENCE_THRESHOLD = 40     # Минимальный приемлемый уровень (≥40%)
BATCH_SIZE = 10  # Количество позиций для параллельной обработки

# OpenAI настройки
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
VECTOR_STORE_ID = os.getenv("VECTOR_STORE_ID")

# Настройки интерфейса
CONFIDENCE_COLORS = {
    "high": "#28a745",      # Зеленый для высокой уверенности (>80%)
    "medium": "#ffc107",    # Желтый для средней уверенности (60-80%)
    "low": "#dc3545"        # Красный для низкой уверенности (<60%)
}

# Колонки для сопоставления данных
REQUIRED_COLUMNS = {
    "product_name": "Наименование товара",
    "description": "Описание (опционально)",
    "category": "Категория (опционально)",
    "brand": "Бренд (опционально)"
}

# Настройки экспорта
EXPORT_FILENAME_PREFIX = "tnved_results"
EXPORT_FORMATS = ["xlsx", "csv"]

# Проверка критических настроек
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY не найден в переменных окружения!")

# Логирование (для отладки)
DEBUG_MODE = os.getenv("DEBUG", "False").lower() == "true" 