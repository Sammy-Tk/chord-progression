
FROM tensorflow/tensorflow:2.10.0

COPY setup.py /setup.py
COPY chords_prog_proj/api chords_prog_proj/api
COPY chords_prog_proj/interface chords_prog_proj/interface
COPY chords_prog_proj/ml_logic chords_prog_proj/ml_logic
COPY chord_to_id.json /chord_to_id.json
COPY mlops /mlops
COPY requirements.txt /requirements.txt

RUN pip install --upgrade pip
RUN pip install -e .

CMD uvicorn chords_prog_proj.api.fast:app --host 0.0.0.0 --port $PORT
