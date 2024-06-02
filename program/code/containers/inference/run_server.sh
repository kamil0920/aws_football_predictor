#!/bin/bash
uvicorn inference:app --proxy-headers --host 0.0.0.0 --port 8080