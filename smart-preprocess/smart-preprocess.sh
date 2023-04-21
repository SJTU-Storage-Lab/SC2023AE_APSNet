#!/bin/bash
SECONDS=0

if [ ! -d "output" ]
then
	mkdir output
fi
if [ ! -d "good" ] 
then
	mkdir good
fi
if [ ! -d "good/mc1" ]
then
        mkdir good/mc1
fi
if [ ! -d "good/mc2" ]
then
        mkdir good/mc2
fi
if [ ! -d "failed" ]
then
	mkdir failed
fi
if [ ! -d "failed/mc1" ]
then
        mkdir failed/mc1
fi
if [ ! -d "failed/mc2" ]
then
        mkdir failed/mc2
fi

g++ parse_smart.cpp -o parse_smart
./parse_smart

python delete_column_mc1.py
python delete_column_mc2.py
python delete_row_mc1.py
python delete_row_mc2.py

g++ -std=c++11 fetch_all_bad.cpp -o failed_all_bad
./failed_all_bad
g++ -std=c++11 failed_single.cpp -o failed_single
./failed_single

g++ -std=c++11 partial_statistics_good_model1.cpp -o partial_statistics_good_model1
./partial_statistics_good_model1

g++ -std=c++11 partial_statistics_good_model2.cpp -o partial_statistics_good_model2
./partial_statistics_good_model2

g++ -std=c++11 statistics_bad_model1.cpp -o statistics_bad_model1
./statistics_bad_model1

g++ -std=c++11 statistics_bad_model2.cpp -o statistics_bad_model2
./statistics_bad_model2

python c2py.py

echo "Elapsed Time: $SECONDS seconds"
echo "Done"
