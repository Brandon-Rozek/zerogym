#!/bin/bash
export FLASK_APP=gymserver.py
for i in {0..31}
do
	python gymserver.py $((5000 + i)) &
done
