#!/usr/bin/bash
echo "delete *.pyc"
find . -name "*.pyc" -delete
echo "delete __pycache__"
rm -rf __pycache__
