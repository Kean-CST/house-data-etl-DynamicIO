[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/TpZXXstf)
# House Sale Data ETL Pipeline

## Overview

In this assignment you will build an ETL (Extract, Transform, Load) pipeline using **PySpark** that processes a house sales dataset and loads it into PostgreSQL, with each neighborhood stored in its own table.

---

## Repository Structure

```
house-sale-etl/
├── dataset/
│   └── historical_purchases_2.csv   # input data (do not modify)
├── src/
│   └── etl_pipeline.py              # your implementation goes here
├── .env.example                     # environment variable template
└── .gitignore
```

---

## Dataset

`dataset/historical_purchases_2.csv` contains house sale records across multiple neighborhoods. It includes all features describing each property and its sale.

---

## Setup

### 1. Install dependencies

Install the following Python packages into your environment:

```
pyspark
python-dotenv
```

### 2. Configure environment variables

Copy `.env.example` to `.env` and fill in your PostgreSQL credentials:

```bash
cp .env.example .env
```

---

## Your Task

Open `src/etl_pipeline.py` and complete the three TODO sections:

- **Extract** – load the CSV into a PySpark DataFrame
- **Transform** – split the data by neighborhood and save each as a separate CSV file
- **Load** – insert each neighborhood's data into its own PostgreSQL table

