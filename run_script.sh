#!/bin/sh
python main.py iris ocpboost random_stump 100
python main.py iris osboost random_stump 100
python main.py iris expboost random_stump 100
python main.py iris smoothboost random_stump 100
python main.py iris ozaboost random_stump 100

python main.py cancer ocpboost random_stump 100
python main.py cancer expboost random_stump 100
python main.py cancer smoothboost random_stump 100
python main.py cancer ozaboost random_stump 100
python main.py cancer osboost random_stump 100

python main.py mushrooms ocpboost random_stump 100
python main.py mushrooms expboost random_stump 100
python main.py mushrooms smoothboost random_stump 100
python main.py mushrooms ozaboost random_stump 100
python main.py mushrooms osboost random_stump 100

python main.py ionosphere ocpboost random_stump 100
python main.py ionosphere expboost random_stump 100
python main.py ionosphere smoothboost random_stump 100
python main.py ionosphere ozaboost random_stump 100
python main.py ionosphere osboost random_stump 100

python main.py heart ocpboost random_stump 100
python main.py heart expboost random_stump 100
python main.py heart smoothboost random_stump 100
python main.py heart ozaboost random_stump 100
python main.py heart osboost random_stump 100