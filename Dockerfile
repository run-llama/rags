FROM python:3.11-slim


RUN apt-get update &&\
    apt-get install -y curl &&\
    pip install --upgrade pip &&\
    curl -sSL https://install.python-poetry.org -o install-poetry.py &&\
    python install-poetry.py &&\
    rm install-poetry.py

ENV PATH /root/.local/bin:$PATH

WORKDIR /app

COPY . /app/

RUN poetry install --with dev

EXPOSE 8501

CMD poetry run streamlit run 1_🏠_Home.py

