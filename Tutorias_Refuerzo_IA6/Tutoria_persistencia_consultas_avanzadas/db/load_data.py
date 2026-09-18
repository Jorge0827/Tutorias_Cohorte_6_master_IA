from pathlib import Path

from connection import ROOT_DIR, get_connection

DATA_DIR = ROOT_DIR / "data"
SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"
EXPECTED_COUNTS = {
    "clientes": 10000,
    "productos": 2000,
    "ventas": 200000,
}
COPY_ORDER = ("clientes", "productos", "ventas")


def apply_schema(cursor):
    schema_sql = SCHEMA_PATH.read_text(encoding="utf-8")
    cursor.execute(schema_sql)


def copy_csv(cursor, table_name):
    csv_path = DATA_DIR / f"{table_name}.csv"
    copy_sql = f"COPY {table_name} FROM STDIN WITH (FORMAT csv, HEADER true)"
    with csv_path.open("r", encoding="utf-8") as csv_file:
        cursor.copy_expert(copy_sql, csv_file)


def validate_counts(cursor):
    results = {}
    for table_name, expected in EXPECTED_COUNTS.items():
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        actual = cursor.fetchone()[0]
        results[table_name] = (actual, expected, actual == expected)
    return results


def main():
    conn = get_connection()
    try:
        with conn:
            with conn.cursor() as cursor:
                apply_schema(cursor)
                for table_name in COPY_ORDER:
                    copy_csv(cursor, table_name)
                counts = validate_counts(cursor)

        print("Carga completada.")
        all_ok = True
        for table_name, (actual, expected, ok) in counts.items():
            status = "OK" if ok else "ERROR"
            print(f"COUNT(*) {table_name}: {actual} (esperado {expected}) [{status}]")
            all_ok = all_ok and ok

        if not all_ok:
            raise SystemExit(1)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
