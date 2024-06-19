# Healthcare Mining: An AI-Powered Approach to Digital Healthcare Data

## Introduction

In the expansive realm of digital healthcare data, individuals encounter significant difficulties in finding accurate, pertinent, and trustworthy information. Conventional search methods often produce results that are overly broad, outdated, or lacking in reliability for making well-informed health-related decisions. The intricate nature of medical terminology exacerbates this challenge, along with the ever-evolving landscape of medical knowledge, where new insights and guidelines continually emerge. "Healthcare Mining" aims to bridge these critical gaps by harnessing the power of artificial intelligence (AI) and machine learning (ML) technologies.

## Problem Statement

Our objective is to develop a system capable of not only comprehending users' nuanced queries but also ensuring the accuracy, credibility, and timeliness of the healthcare information it provides.

## System Design and Development

### Overview

Our system architecture is streamlined, focusing on the frontend application developed with Streamlit, which directly interfaces with various LLM APIs and models including OpenAI, Claude, Gemini, Mistral, and LLama2.

#### UI Framework

- **Streamlit**: Facilitates the creation of an interactive user interface, allowing users to input queries, upload files, and receive information in real-time.

### System Architecture and Algorithms

- **Direct Calls to LLM**: The query embedding generation is performed through direct calls to LLM APIs and models. This approach ensures efficient handling of user requests and leverages different LLM’s powerful natural language processing capabilities.

## Different LLMs

We have tested the capabilities of our healthcare chatbot among the most popular LLMs in the market at the moment such as OpenAI, Gemini, Llama, Mistral, and Claude. With each of our responses evaluated, we have figured out the best performing LLM for our chatbot which gives out the most accurate results.

## Trulens as an Evaluation Metric

We have integrated Trulens as our evaluation metric by coding specific feedback functions to measure the responses of our chatbot. This was done across different chains as well as different LLMs to ensure our chatbot performs to the best of its capabilities.

## Data Collection Techniques

- **Web Scraping**: Used the Beautiful Soup library in Python along with the requests library to scrape data from WebMD and Mayo Clinic.

### Steps for Cleaning Scraped HTML Data

1. Clean HTML Tags
2. Strip Whitespace
3. Convert Data Types
4. Standardize Values

## Evaluations

### Technical Performance Metrics

- **Trulens Analysis**: Assesses Groundedness, Questions/Answer Relevance, and Question/Context Relevance. This analysis allows us to gauge how well the system’s responses are rooted in the factual content of our database and their relevance to the user’s queries.

## Installation and Setup

### Requirements

- Python 3.8 or higher
- Streamlit
- Beautiful Soup (for web scrapping)
- Requests Library (for web scrapping)

### Setup Instructions

1. Clone the repository:
    ```bash
    git clone https://github.com/Sujithrt/healthcare_mining.git
    ```
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Obtain the API keys for OpenAI, Claude, and Gemini and save them in a `streamlit/secrets.toml` file
4. Create an account in Datastax Astra and create a database. Obtain the database ID and Token and save them in the `streamlit/secrets.toml` file
5. Download the pre-trained models (Mistral and LLama2) from the below links into the models folder:
    - Mistral: [Download](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/blob/main/mistral-7b-instruct-v0.2.Q8_0.gguf)
    - LLama2: [Download](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/blob/main/llama-2-7b-chat.Q8_0.gguf)
6. Run the application:
    ```bash
    streamlit run OpenAI.py
    ```
