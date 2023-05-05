#!/bin/bash

os=""

uname -a | grep -i Darwin
if [ $? = 0 ];then
  os="mac"
fi

for i in $(find test | grep 'png')
do
  if [ "${os}" = "mac" ];then
    data=$(base64 -b 0 -i $i)
  else
    data=$(base64 -w 0 $i)
  fi

	echo $i
  json="{\"image\":\"${data}\"}"
  curl -H 'Content-Type:application/json' -d "${json}" http://127.0.0.1:8886/predict
  echo
  echo '--------'
done

