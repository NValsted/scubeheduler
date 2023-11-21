FROM python:3.9-buster as builder

RUN pip install poetry==1.4.2

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

WORKDIR /scubeheduler

COPY pyproject.toml poetry.lock ./

RUN poetry install --without dev --no-root && rm -rf $POETRY_CACHE_DIR

FROM python:3.9-slim-buster as runtime

ENV VIRTUAL_ENV=/scubeheduler/.venv \
    PATH="/scubeheduler/.venv/bin:$PATH"

COPY --from=builder ${VIRTUAL_ENV} ${VIRTUAL_ENV}

COPY lib ./lib
COPY scripts ./scripts

ENTRYPOINT ["python", "scripts/main.py"]
