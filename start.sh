#!/bin/sh
python3 flow_parser.py
jina flow --uses flow.tmp.yml
