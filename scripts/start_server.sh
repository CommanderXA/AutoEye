#!/bin/bash

uvicorn app:app --app-dir ./ --host 0.0.0.0 --port 8000 --reload