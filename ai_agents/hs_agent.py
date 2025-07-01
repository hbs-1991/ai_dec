"""
AI DECLARANT - HS Code Agent
Адаптированный агент для определения кодов ТН ВЭД на основе openai-agents
"""

import asyncio
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from dataclasses import dataclass
import config

from agents import (
    Agent, 
    Runner, 
    FileSearchTool, 
    function_tool,
    RunContextWrapper,
    ModelSettings,
    WebSearchTool
)


@dataclass
class ProcessingContext:
    """Контекст обработки для передачи данных между инструментами"""
    batch_id: str
    total_items: int
    processed_items: int = 0
    failed_items: List[str] = None
    
    def __post_init__(self):
        if self.failed_items is None:
            self.failed_items = []


class HSCodeSearchResult(BaseModel):
    """Структурированный результат поиска ТН ВЭД кода"""
    hs_code: str = Field(description="Найденный 9-значный код ТН ВЭД (например: 8517.12.000)")
    confidence: int = Field(ge=0, le=100, description="Уровень доверия от 0 до 100 (в процентах)")
    description: str = Field(description="Описание товара согласно ТН ВЭД")
    reasoning: str = Field(description="Обоснование выбора кода")
    alternative_codes: List[str] = Field(
        default_factory=list, 
        description="Альтернативные коды при низком уровне доверия"
    )


class BatchProcessingResult(BaseModel):
    """Результат пакетной обработки"""
    results: List[HSCodeSearchResult]
    processing_stats: Dict[str, Any]
    errors: List[str] = Field(default_factory=list)


@function_tool
async def validate_product_description(
    ctx: RunContextWrapper[ProcessingContext], 
    product_name: str
) -> str:
    """Валидирует и нормализует описание товара для лучшего поиска"""
    if not product_name or len(product_name.strip()) < 3:
        return "❌ Описание товара слишком короткое для анализа"
    
    # Простая нормализация
    normalized = product_name.strip().lower()
    
    # Увеличиваем счетчик обработанных элементов
    ctx.context.processed_items += 1
    
    return f"✅ Описание нормализовано: {normalized}"


class HSCodeAgent:
    """Главный класс агента для определения ТН ВЭД кодов"""
    
    def __init__(self, vector_store_id: Optional[str] = None):
        """
        Инициализация агента
        
        Args:
            vector_store_id: ID векторного хранилища с кодами ТН ВЭД
        """
        self.vector_store_id = vector_store_id or config.VECTOR_STORE_ID
        self.confidence_threshold = config.DEFAULT_CONFIDENCE_THRESHOLD
        
        # Создаем агента с FileSearchTool для поиска в векторной базе
        self.agent = Agent(
            name="HS Code Classifier",
            instructions=self._build_instructions(),
            model="gpt-4.1",  # Используем корректное название модели
            model_settings=ModelSettings(temperature=0.1),  # Низкая температура для консистентности
            tools=[
                FileSearchTool(
                    max_num_results=5,
                    vector_store_ids=[self.vector_store_id]
                ),
                WebSearchTool(),
                validate_product_description
            ],
            output_type=HSCodeSearchResult  # Структурированный вывод
        )
    
    def _build_instructions(self) -> str:
        """Создает инструкции для агента"""
        return f"""
Ты - эксперт по Товарной номенклатуре внешнеэкономической деятельности (ТН ВЭД) Туркменистана.

ТВОЯ ЗАДАЧА:
1. Анализировать описания товаров
2. Определять точный код ТН ВЭД (10-значный)
3. Обосновывать выбор кода
4. Указывать уровень доверия (confidence)

ТРЕБОВАНИЯ К АНАЛИЗУ:
- Используй FileSearchTool для поиска по векторной базе кодов ТН ВЭД
- Если в базе нет кода, то ищи в интернете используя WebSearchTool.
- Сначала валидируй описание товара с помощью validate_product_description
- Анализируй материал, назначение, конструкцию товара
- Учитывай особенности классификации Туркменистана
- Confidence ≥ {self.confidence_threshold}% считается высоким
- При низком confidence предложи альтернативные коды

ФОРМАТ ВЫВОДА - СТРОГО HSCodeSearchResult:
- hs_code: точный 9-значный код (например: "8517.12.000")
- confidence: от 0 до 100 (в процентах, где 100 = полная уверенность)
- description: официальное описание согласно ТН ВЭД
- reasoning: подробное обоснование выбора
- alternative_codes: список альтернатив при uncertainty

ПРИМЕРЫ ХОРОШЕГО АНАЛИЗА:
✅ "Смартфон Apple iPhone" → "8517.12.000" (телефоны сотовой связи, confidence: 95)
✅ "Кофе в зернах арабика" → "0901.11.000" (кофе необжаренный, confidence: 90)
✅ "Автомобильные шины R16" → "4011.10.000" (шины новые для легковых авто, confidence: 85)

Будь точным, последовательным и обоснованным в каждом решении!
"""

    async def classify_single_item(
        self, 
        product_description: str, 
        additional_info: Optional[Dict[str, str]] = None
    ) -> HSCodeSearchResult:
        """
        Классифицирует один товар
        
        Args:
            product_description: Описание товара
            additional_info: Дополнительная информация (бренд, категория и т.д.)
            
        Returns:
            HSCodeSearchResult с результатом классификации
        """
        # Формируем расширенное описание
        full_description = product_description
        if additional_info:
            additional_parts = []
            for key, value in additional_info.items():
                if value and value.strip():
                    additional_parts.append(f"{key}: {value}")
            if additional_parts:
                full_description += f" ({', '.join(additional_parts)})"
        
        # Создаем контекст обработки
        context = ProcessingContext(
            batch_id="single_item",
            total_items=1
        )
        
        try:
            # Выполняем классификацию
            result = await Runner.run(
                starting_agent=self.agent,
                input=f"Определи код ТН ВЭД для товара: {full_description}",
                context=context,
                max_turns=10  # Ограничиваем количество итераций
            )
            
            # Получаем структурированный результат
            classification = result.final_output
            
            # Валидируем результат
            if not isinstance(classification, HSCodeSearchResult):
                # Фоллбэк для случаев, когда структурированный вывод не сработал
                return HSCodeSearchResult(
                    hs_code="0000.00.000",
                    confidence=10,
                    description="Ошибка обработки - не удалось получить структурированный результат",
                    reasoning=f"Агент вернул: {str(result.final_output)}"
                )
            
            return classification
            
        except Exception as e:
            # Возвращаем результат с ошибкой
            return HSCodeSearchResult(
                hs_code="0000.00.000",
                confidence=0,
                description="Ошибка классификации",
                reasoning=f"Произошла ошибка при обработке: {str(e)}"
            )

    async def classify_batch(
        self, 
        items: List[Dict[str, str]], 
        progress_callback: Optional[callable] = None
    ) -> BatchProcessingResult:
        """
        Классифицирует пакет товаров с поддержкой прогресса
        
        Args:
            items: Список словарей с описаниями товаров
            progress_callback: Функция для отслеживания прогресса
            
        Returns:
            BatchProcessingResult с результатами обработки
        """
        print(f"🚀 [DEBUG] classify_batch начат с {len(items)} элементами")
        
        results = []
        errors = []
        batch_size = config.BATCH_SIZE
        
        print(f"📦 [DEBUG] Размер батча: {batch_size}")
        
        # Создаем контекст для пакетной обработки
        context = ProcessingContext(
            batch_id=f"batch_{len(items)}",
            total_items=len(items)
        )
        
        # Обрабатываем по пакетам для контроля производительности
        total_batches = (len(items) + batch_size - 1) // batch_size
        print(f"🔢 [DEBUG] Всего батчей для обработки: {total_batches}")
        
        for batch_index, i in enumerate(range(0, len(items), batch_size)):
            batch = items[i:i + batch_size]
            print(f"🎯 [DEBUG] Обрабатываем батч {batch_index + 1}/{total_batches} с {len(batch)} элементами")
            
            # Создаем задачи для параллельной обработки
            tasks = []
            for item_index, item in enumerate(batch):
                # Извлекаем основное описание и дополнительную информацию
                main_description = item.get('product_name', '') or item.get('description', '')
                additional_info = {k: v for k, v in item.items() 
                                 if k not in ['product_name', 'description', 'row_index'] and v}
                
                print(f"  📝 [DEBUG] Элемент {item_index + 1}: '{main_description}' + доп.инфо: {additional_info}")
                
                task = self.classify_single_item(main_description, additional_info)
                tasks.append(task)
            
            print(f"⚡ [DEBUG] Создано {len(tasks)} задач для параллельного выполнения")
            
            # Выполняем пакет параллельно
            try:
                print(f"🔄 [DEBUG] Запускаем asyncio.gather для {len(tasks)} задач...")
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                print(f"✅ [DEBUG] asyncio.gather завершен, получено {len(batch_results)} результатов")
                
                for j, result in enumerate(batch_results):
                    if isinstance(result, Exception):
                        error_msg = f"Ошибка обработки элемента {i+j+1}: {str(result)}"
                        print(f"❌ [DEBUG] {error_msg}")
                        errors.append(error_msg)
                        # Добавляем заглушку результата
                        results.append(HSCodeSearchResult(
                            hs_code="0000.00.000",
                            confidence=0,
                            description="Ошибка обработки",
                            reasoning=f"Исключение: {str(result)}"
                        ))
                    else:
                        print(f"✅ [DEBUG] Элемент {i+j+1} обработан: код={result.hs_code}, доверие={result.confidence}%")
                        results.append(result)
                
                # Вызываем callback для прогресса
                if progress_callback:
                    processed_count = min(i + batch_size, len(items))
                    print(f"📊 [DEBUG] Обновляем прогресс: {processed_count}/{len(items)}")
                    try:
                        progress_callback(processed_count, len(items))
                    except Exception as callback_error:
                        print(f"⚠️ [DEBUG] Ошибка в progress_callback: {callback_error}")
                        
            except Exception as e:
                error_msg = f"Ошибка обработки пакета {batch_index + 1}: {str(e)}"
                print(f"💥 [DEBUG] {error_msg}")
                errors.append(error_msg)
                
                # Добавляем заглушки для всего пакета
                for item in batch:
                    results.append(HSCodeSearchResult(
                        hs_code="0000.00.000",
                        confidence=0,
                        description="Ошибка пакетной обработки",
                        reasoning=error_msg
                    ))
        
        # Собираем статистику
        successful_classifications = [r for r in results if r.confidence > 0]
        high_confidence = [r for r in successful_classifications 
                          if r.confidence >= self.confidence_threshold]
        
        processing_stats = {
            "total_items": len(items),
            "successful_classifications": len(successful_classifications),
            "high_confidence_results": len(high_confidence),
            "average_confidence": (
                sum(r.confidence for r in successful_classifications) / len(successful_classifications)
                if successful_classifications else 0.0
            ),
            "processing_errors": len(errors)
        }
        
        print(f"📈 [DEBUG] Итоговая статистика: {processing_stats}")
        print(f"🏁 [DEBUG] classify_batch завершен с {len(results)} результатами")
        
        return BatchProcessingResult(
            results=results,
            processing_stats=processing_stats,
            errors=errors
        )

    def get_confidence_color(self, confidence: int) -> str:
        """Возвращает цвет для отображения уровня доверия"""
        if confidence >= 80:
            return config.CONFIDENCE_COLORS["high"]
        elif confidence >= 60:
            return config.CONFIDENCE_COLORS["medium"] 
        else:
            return config.CONFIDENCE_COLORS["low"]


# Удобная функция для быстрого использования
async def classify_hs_code(
    product_description: str, 
    vector_store_id: Optional[str] = None
) -> HSCodeSearchResult:
    """
    Быстрая функция для классификации одного товара
    
    Args:
        product_description: Описание товара
        vector_store_id: ID векторного хранилища (опционально)
        
    Returns:
        HSCodeSearchResult с кодом ТН ВЭД
    """
    agent = HSCodeAgent(vector_store_id)
    return await agent.classify_single_item(product_description)


# Экспортируем основные классы
__all__ = [
    'HSCodeAgent',
    'HSCodeSearchResult', 
    'BatchProcessingResult',
    'ProcessingContext',
    'classify_hs_code'
] 