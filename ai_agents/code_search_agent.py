import os
from dotenv import load_dotenv
from agents import Agent, FileSearchTool, Runner
import asyncio
from pydantic import BaseModel

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class HSCodeSearchResult(BaseModel):
    product_name: str
    hs_code: str
    code_description: str
    confidence: float
    

hs_code_search_agent = Agent(
    name="HS Code Search Agent",
    instructions="Роль: Ты — высокоточный классификатор товаров по ТН ВЭД Туркменистана, использующий векторную базу описаний в котором загружена ТН ВЭД кодовая база Туркменистана."
                  "Задание: По каждому входному описанию товара определяй наиболее релевантный 9-значный код ТН ВЭД."
                  "Правила ответа:"
                  "• Выводи только сам код (9 цифр) без каких-либо символов, текста или пояснений."
                  "Формат ввода: произвольное текстовое описание товара (RU)."
                  "Формат вывода: `HSCodeSearchResult`.",
    model= "gpt-4.1",
    tools= [
        FileSearchTool(
            vector_store_ids= ["vs_685e39e636b4819192a076ca914b0e1e"],
            max_num_results= 3,
            include_search_results= True
        )
    ],
    output_type=HSCodeSearchResult
)

async def main():
    result = await Runner.run(hs_code_search_agent, "Какой ТНВЭД код [Наименование товара :алюминевый профиль]")
    print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())




