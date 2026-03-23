"""
House Sale Data ETL Pipeline
============================
Implement the three functions below to complete the ETL pipeline.

Steps:
  1. EXTRACT  – load the CSV into a PySpark DataFrame
  2. TRANSFORM – split the data by neighborhood and save each as a separate CSV
  3. LOAD      – insert each neighborhood DataFrame into its own PostgreSQL table
"""
from __future__ import annotations

import csv  # noqa: F401
import os  # noqa: F401
from pathlib import Path

from dotenv import load_dotenv  # noqa: F401
from pyspark.sql import DataFrame, SparkSession  # noqa: F401
from pyspark.sql import functions as F  # noqa: F401

# ── Predefined constants (do not modify) ──────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent

NEIGHBORHOODS = [
    "Downtown", "Green Valley", "Hillcrest", "Lakeside", "Maple Heights",
    "Oakwood", "Old Town", "Riverside", "Suburban Park", "University District",
]

OUTPUT_DIR   = ROOT / "output" / "by_neighborhood"
OUTPUT_FILES = {hood: OUTPUT_DIR / f"{hood.replace(' ', '_').lower()}.csv" for hood in NEIGHBORHOODS}

PG_TABLES = {hood: f"public.{hood.replace(' ', '_').lower()}" for hood in NEIGHBORHOODS}

PG_COLUMN_SCHEMA = (
    "house_id TEXT, neighborhood TEXT, price INTEGER, square_feet INTEGER, "
    "num_bedrooms INTEGER, num_bathrooms INTEGER, house_age INTEGER, "
    "garage_spaces INTEGER, lot_size_acres NUMERIC(6,2), has_pool BOOLEAN, "
    "recently_renovated BOOLEAN, energy_rating TEXT, location_score INTEGER, "
    "school_rating INTEGER, crime_rate INTEGER, "
    "distance_downtown_miles NUMERIC(6,2), sale_date DATE, days_on_market INTEGER"
)


_BOOL_COLS = ("has_pool", "recently_renovated", "has_children", "first_time_buyer")

_PG_COLS = [
    "house_id", "neighborhood", "price", "square_feet", "num_bedrooms",
    "num_bathrooms", "house_age", "garage_spaces", "lot_size_acres",
    "has_pool", "recently_renovated", "energy_rating", "location_score",
    "school_rating", "crime_rate", "distance_downtown_miles",
    "sale_date", "days_on_market",
]


def extract(spark: SparkSession, csv_path: str) -> DataFrame:
    """Load the CSV dataset into a PySpark DataFrame with correct data types."""
    df = spark.read.option("header", "true").csv(csv_path)

    # Normalise sale_date to ISO format (source is M/d/yy)
    df = df.withColumn(
        "sale_date",
        F.date_format(F.to_date(F.col("sale_date"), "M/d/yy"), "yyyy-MM-dd"),
    )

    # Normalise boolean columns to Python title-case strings ("True" / "False")
    for col in _BOOL_COLS:
        df = df.withColumn(
            col,
            F.when(F.upper(F.col(col)) == "TRUE", F.lit("True")).otherwise(F.lit("False")),
        )

    return df


def transform(df: DataFrame) -> dict[str, DataFrame]:
    """Split the data by neighborhood and save each as a separate CSV file."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    partitions: dict[str, DataFrame] = {}
    for hood in NEIGHBORHOODS:
        hood_df = df.filter(F.col("neighborhood") == hood).orderBy("house_id")
        partitions[hood] = hood_df

        # toPandas preserves exact string values so numeric formatting is unchanged
        hood_df.toPandas().to_csv(OUTPUT_FILES[hood], index=False)

    return partitions


def load(partitions: dict[str, DataFrame], jdbc_url: str, pg_props: dict) -> None:
    """Insert each neighborhood dataset into its own PostgreSQL table."""
    from pyspark.sql.types import BooleanType, DoubleType, IntegerType

    int_cols = {
        "price", "square_feet", "num_bedrooms", "num_bathrooms",
        "house_age", "garage_spaces", "location_score", "school_rating",
        "crime_rate", "days_on_market",
    }

    for hood, df in partitions.items():
        pg_df = df.select(_PG_COLS)

        for c in int_cols:
            pg_df = pg_df.withColumn(c, F.col(c).cast(IntegerType()))

        for c in ("lot_size_acres", "distance_downtown_miles"):
            pg_df = pg_df.withColumn(c, F.col(c).cast(DoubleType()))

        for c in ("has_pool", "recently_renovated"):
            pg_df = pg_df.withColumn(c, F.col(c).cast(BooleanType()))

        pg_df = pg_df.withColumn("sale_date", F.to_date(F.col("sale_date"), "yyyy-MM-dd"))

        pg_df.write.jdbc(
            url=jdbc_url,
            table=PG_TABLES[hood],
            mode="overwrite",
            properties=pg_props,
        )


# ── Main (do not modify) ───────────────────────────────────────────────────────
def main() -> None:
    load_dotenv(ROOT / ".env")

    jdbc_url = (
        f"jdbc:postgresql://{os.getenv('PG_HOST', 'localhost')}:"
        f"{os.getenv('PG_PORT', '5432')}/{os.environ['PG_DATABASE']}"
    )
    pg_props = {
        "user":     os.environ["PG_USER"],
        "password": os.getenv("PG_PASSWORD", ""),
        "driver":   "org.postgresql.Driver",
    }
    csv_path = str(ROOT / os.getenv("DATASET_DIR", "dataset") / os.getenv("DATASET_FILE", "historical_purchases.csv"))

    spark = (
        SparkSession.builder.appName("HouseSaleETL")
        .config("spark.jars.packages", "org.postgresql:postgresql:42.7.3")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    df         = extract(spark, csv_path)
    partitions = transform(df)
    load(partitions, jdbc_url, pg_props)

    spark.stop()


if __name__ == "__main__":
    main()
