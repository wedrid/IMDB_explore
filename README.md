# IMDB_explore
Some explorations of the IMDB movie dataset using graph algorithms.
The processing cover the following questions: 
1.G) Considering only the movies up to year x with x in {1930,1940,1950,1960,1970,1980,1990,2000,2010,2020}, write a function which, given x, computes the average number of movies per actor up to year x.
2.3) Considering only the movies up to year x with x in {1930,1940,1950,1960,1970,1980,1990,2000,2010,2020} and restricting to the largest connected component of the graph. Approximate the closeness centrality for each node. Who are the top-10 actors?
3.III) Which is the pair of movies that share the largest number of actors?
4.-) Which is the pair of actors who collaborated the most among themselves?

This project was part of the exam of advanced algorithms and graph mining of the MSc degree in computing engineering of the University of Florence. 

The python packages used can be found in the requirements.txt

For precalculated output it is suggested to see the file project.ipynb, which was used as workbench and in which results are for the most part saved.

For the execution, a .py was exported with the jupyter functionality. 
The flag q4 allows the user to execute either "question 4" or the other questions; this is because for question 4 another graph is created, and RAM issues can occur. 
