FROM python:3.11

COPY requirements.txt /workdir/
COPY myapp/ /workdir/myapp/
COPY static/ /workdir/static/
COPY models/ /workdir/models/
COPY checkpoints/ /workdir/checkpoints/
COPY emnist-balanced-mapping.txt /workdir/

WORKDIR /workdir

RUN pip install -r requirements.txt
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# Run the application
CMD ["uvicorn", "myapp.main:app", "--host", "0.0.0.0", "--port", "8000"]
