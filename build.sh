#!/bin/bash

pip install -r requirements.txt
cd frontend
npm run build
#npm install && npm run build
#pwd
cd ..
uvicorn backend.main:app --reload
