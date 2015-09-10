#!/bin/bash

FNAME=assignment1.md

### Set initial time of file
LTIME=`stat -c %Z $FNAME`

while true    
do
   ATIME=`stat -c %Z $FNAME`

   if [[ "$ATIME" != "$LTIME" ]]
   then    
       echo "RUN COMMNAD"
       # pandoc assignment1.md -s --mathjax -o assignment1.html
       pandoc assignment1.md -s  --mathml -o assignment1.html
       LTIME=$ATIME
   fi
   sleep 1
done

