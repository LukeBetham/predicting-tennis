# Predicting Tennis - Modeling Match Outcomes with Machine Learning
### General Assembly Capstone Project

How accurately can you predict who will win a tennis match given past data? 
In this project I use data from 30 years of tennis and around 75,000 professional individual matches to predict who is the likely winner of any given tennis match. 

## Goals
My goals for this project were:
- Gather usable data for Tennis matches, players and bookmaker odds
- Clean and analyse the data, making sure that it is reliable and workable
- Feature Creation - use the data to create features which will feed and improve the model.
- Train Machine Learning models, evaluate predicitons and look at potential new features on the back of results. 
- Prediction Goal Primary Objective - Beat ranking baseline - create a model that will be able to predict the outcome of a tennis match better than just choosing the highest ranked player.
- Prediction Goal Secondary Objective - Beat the bookmakers predictions.

## The Data
For my project I will mainly be using data extracted from a docker container (https://github.com/mcekovic/tennis-crystal-ball/issues/337) - which was created opn the back of a large open source atp data repository from github = https://github.com/JeffSackmann/tennis_atp

I checked that this data was valid by cross referencing a number of stats between this database and the offcial ATP website, and everything I checked matched up. I will be using the last 30 years of data, as this is when the ATP started recording detailed match data, and so before this the information isnt as good. See below for visual representation of the usable matches. I only used data from 1991 as this is when the detailed data started.

In addition to this I have sourced some data files which include the betting odds for the last 20 years in order to check how my model measures up. I got this info from http://www.tennis-data.co.uk/alldata.php

Some issues I ran into in the data collection stage:
- Docker Dependencies - as the docker image was built to serve the ultimate tennis website, the sql database downloaded from the docker image had a number of dependencies which meant I had to manually review the raw sql file and remove all dependencies before I could load the database. 
- WTA Data incomplete - I was not able to get the same quality and range of data for women's tennis as men's tennis and so unfortunately for the time being I only focused on the data for men's tennis. I will adapt for another project in the future!
- Commonality Issues: There were no data fields in common which could link the betting odds data to exact matches and so I had to do some feature engineering to link the odds to any given match.  
![](images/data_not_full.png)


## EDA
Before starting on the modeling I did some analysis of the data to see if any specific patterns emerged
![](images/age.png)
![](images/gamestyle.png)
![](images/matches2019_2.gif)

## Predictive Models and Evaluation

## Odds Comparison

## Further Analysis


Over the last 
My capstone project from the General Assembly immersive program. Here I predicted outcomes of tennis matches. I extracted the data using SQL and created symmetrical models using logistic regression, gradient boosters and neural nets. Using AWS infrastructure, I tuned the final model to beat the bookmakers. 

