from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlmodel import SQLModel, Session
from typing import Annotated
from fastapi import Depends

DATABASE_URL = "postgresql+psycopg2://postgres:getlucky15@localhost:5432/postgres"
engine = create_engine(DATABASE_URL, echo=True)

Base = declarative_base()

SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)




# Code above omitted 👆

def get_session():
    with Session(engine) as session:
        yield session


# SessionDep = Annotated[Session, Depends(get_session)]

# Code below omitted 👇