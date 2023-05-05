#!/bin/bash

for i in $(find test | grep 'png')
do
	echo $i
	curl -X POST -F "image=@${i}" http://127.0.0.1:8886/predict
	echo
	echo '--------------'
done