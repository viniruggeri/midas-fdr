from sqlalchemy import create_engine, MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
import cx_Oracle

from config import settings

# PostgreSQL
POSTGRES_URL = f"postgresql://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"

postgres_engine = create_engine(
    POSTGRES_URL,
    poolclass=StaticPool,
    echo=settings.DEBUG
)

PostgresSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=postgres_engine)
postgres_metadata = MetaData()

# Oracle Database (
ORACLE_URL = f"oracle+cx_oracle://{settings.ORACLE_USER}:{settings.ORACLE_PASSWORD}@{settings.ORACLE_HOST}:{settings.ORACLE_PORT}/{settings.ORACLE_SERVICE}"

oracle_engine = create_engine(
    ORACLE_URL,
    echo=settings.DEBUG
)

OracleSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=oracle_engine)
oracle_metadata = MetaData()

Base = declarative_base()


def get_postgres_db():
    db = PostgresSessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_oracle_db():
    db = OracleSessionLocal()
    try:
        yield db
    finally:
        db.close()


async def init_postgres_tables():
    postgres_metadata.create_all(bind=postgres_engine)


async def init_oracle_connection():
    try:
        with oracle_engine.connect() as conn:
            result = conn.execute("SELECT 1 FROM DUAL")
            return True
    except Exception as e:
        print(f"Oracle connection failed: {e}")
        return False