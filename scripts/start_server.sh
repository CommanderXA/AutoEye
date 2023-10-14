#!/bin/bash

uvicorn api:app --app-dir ./server/ --host 0.0.0.0 --port 8000 --reload