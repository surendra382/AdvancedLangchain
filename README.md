# Advanced Chains in LangChain

## Table of Contents
1. [Introduction](#introduction)
2. [LangSmith](#langsmith)
   - [What is LangSmith?](#what-is-langsmith)
   - [Why is LangSmith Important?](#why-is-langsmith-important)
   - [Setting Up LangSmith](#setting-up-langsmith)
3. [Functions Inside Chains](#functions-inside-chains)
   - [What is Coercion?](#what-is-coercion)
   - [Coercing Functions into Chains](#coercing-functions-into-chains)
4. [Runnable Parallels](#runnable-parallels)
   - [Running Multiple Functions in Parallel](#running-multiple-functions-in-parallel)
   - [Running Multiple Chains in Parallel](#running-multiple-chains-in-parallel)
5. [Runnable Pass-Through](#runnable-pass-through)
   - [Concept Overview](#concept-overview)
   - [Transforming Data with RunnablePassthrough](#transforming-data-with-runnablepassthrough)
6. [Final Exercise: Conversational Assistant with Memory](#final-exercise-conversational-assistant-with-memory)

---

## Introduction
Welcome to the fourth module of our course on **LangChain Advanced Chains**! We’ve covered the basics of Generative AI, software architecture, and LangChain’s foundational features. Now, we will dive into advanced chaining techniques to create more complex and intelligent AI workflows.

### Topics Covered in This Module:
1. **LangSmith Introduction** - Learn about LangSmith, an observability tool for tracking LangChain applications.
2. **Coercing Functions into Chains** - Convert functions into LangChain-compatible runnables.
3. **Runnable Parallels** - Execute multiple chains in parallel for improved efficiency.
4. **Runnable Pass-Through** - Maintain data across chain steps without modifications.
5. **Memory in Chains** - Implement memory in your conversational AI assistant.

By the end of this module, you’ll be able to build a **conversational assistant with memory** that can track and analyze complex interactions.

---

## LangSmith

### What is LangSmith?
LangSmith is an observability tool created by the developers of LangChain. It provides visibility into your LLM-powered applications, tracking:
- 📡 Requests to LLMs, whether third-party APIs (OpenAI, Anthropic, etc.) or local models.
- 🔄 Data flow within chains to monitor variable passing and output generation.
- ⚡ Parallel operations to optimize performance.
- 🔢 Token usage to track costs and efficiency.

### Why is LangSmith Important?
Building complex workflows in LangChain means managing multiple API calls, nested chains, and concurrent operations. LangSmith offers **real-time tracking** of each step, making debugging and optimization much easier.

### Setting Up LangSmith
To integrate LangSmith with LangChain, follow these steps:
#### Step 1: Create Your LangSmith Account and API Key
1. Sign up on the LangSmith dashboard.
2. Navigate to **API Keys** and create a new key.
3. Copy the key for later use.

#### Step 2: Configure Your Environment
Add the following environment variables:
```python
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "<YOUR_API_KEY_HERE>"
```

#### Step 3: Start Logging Your Application
Simply run your LangChain code, and LangSmith will automatically log activities:
```python
from langchain_groq import ChatGroq
llm = ChatGroq()
result = llm.invoke("Write a poem about the stars.")
print(result)
```

---

## Functions Inside Chains

### What is Coercion?
Coercion allows you to **convert regular functions into LangChain runnables**, making them compatible with chains. This is particularly useful when integrating APIs or processing data outside the LLM.

### Coercing Functions into Chains
Example: Function coercion using a **lambda function**:
```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser

def get_weather_of(city):
    return "rainy" if city == "London" else "cloudy"

prompt = ChatPromptTemplate.from_template("What should I wear if the weather today is {weather}?")
model = ChatGroq()
output_parser = StrOutputParser()

weather_chain = (
    (lambda input: get_weather_of(input)) | prompt | model | output_parser
)
result = weather_chain.invoke("London")
print(result)
```

---

## Runnable Parallels

### Running Multiple Functions in Parallel
Runnable Parallels allow running independent functions simultaneously:
```python
def get_temperature(city):
    return 15 if city == "London" else 25

def get_weather(city):
    return "rainy" if city == "London" else "sunny"

from langchain_core.runnables import RunnableParallel
parallel_chain = RunnableParallel(
    temperature=get_temperature, weather=get_weather
)
result = parallel_chain.invoke("London")
print(result)
```

### Running Multiple Chains in Parallel
Run two different chains concurrently and merge their outputs:
```python
from langchain_core.prompts import ChatPromptTemplate
prompt_template = "Write a poem about {topic}"
prompt = ChatPromptTemplate.from_template(prompt_template)
model = ChatGroq()
output_parser = StrOutputParser()
poem_chain = prompt | model | output_parser

comparison_template = "Compare the following:
A poem: {poem}
A prose: {prose}"
comparison_prompt = ChatPromptTemplate.from_template(comparison_template)
comparison_chain = comparison_prompt | model | output_parser

composed_chain = (
    RunnableParallel(poem=poem_chain, prose=poem_chain) | comparison_chain
)
result = composed_chain.invoke({"topic": "coffee"})
print(result)
```

---

## Runnable Pass-Through

### Concept Overview
Runnable Pass-Through allows maintaining values between steps without modifications.

Example: Passing query data without alteration:
```python
from langchain_core.runnables import RunnablePassthrough
retrieval_chain = (
    {"context": retriever, "question": RunnablePassthrough()} |
    prompt | model | StrOutputParser()
)
result = retrieval_chain.invoke("Who was the funniest Beatle?")
print(result)
```

### Transforming Data with RunnablePassthrough
Assign new keys while passing data forward:
```python
from langchain_core.runnables import RunnablePassthrough
chain = (
    {"temperature_c": lambda input: 20, "city": RunnablePassthrough()} |
    RunnablePassthrough.assign(temperature_f=lambda inputs: (inputs["temperature_c"] * 9/5) + 32)
)
print(chain.invoke("Paris"))
```

---

## Final Exercise: Conversational Assistant with Memory

**Goal:** Build a conversational AI assistant using LangChain advanced chaining techniques.

### Steps:
1. Integrate **LangSmith** for real-time observability.
2. Use **coercion** to integrate external API calls.
3. Leverage **Runnable Parallels** for concurrent processing.
4. Implement **Runnable Pass-Through** for maintaining conversation context.

By completing this exercise, you’ll have a powerful AI assistant capable of handling dynamic interactions while maintaining context!

---

### 🎯 Congratulations!
You’ve successfully explored advanced LangChain techniques. Now, go ahead and **experiment** by customizing and optimizing your AI workflows!

For more details, visit [LangChain Documentation](https://docs.langchain.com). 🚀

