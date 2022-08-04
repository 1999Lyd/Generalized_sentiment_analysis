install: 
	pip install --upgrade pip &&\
		pip install -r requirements.txt

format:
	black *.py

lint:
	pylint --disable=E1101,W0123,C0301,C0103,C0116 app.py
