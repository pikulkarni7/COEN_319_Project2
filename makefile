CC = g++

default:  sparsematmult

sparsematmult: sparsematmult.cpp
	${CC} -O3 -Wall -Wextra -fopenmp -o $@ sparsematmult.cpp


clean:
	
	-rm -f sparsematmult
