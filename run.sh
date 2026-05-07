#!/bin/bash
set -e

echo "==> Installing Python dependencies..."
pip install -r requirements.txt --quiet --break-system-packages

echo "==> Installing Node dependencies and building frontend..."
cd src/bot/client/project
npm install --silent
npm run build
cd ../../../..

echo "==> Starting server..."
cd src/bot/server
python hindi_bot.py
