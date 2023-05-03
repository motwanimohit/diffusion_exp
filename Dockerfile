FROM python:3.10-slim-buster

WORKDIR /app

#make a virtual environment
ENV VIRTUAL_ENV=/app/env/
RUN python -m venv $VIRTUAL_ENV
# set path to virtual environment
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY requirements.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY diffusion/ diffusion/
COPY main.py /app/main.py

CMD ["python", "main.py"]