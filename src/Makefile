CC := g++ -I ./libs/Eigen/
CFLAGS = -Wall -g -O3 

.PHONY: all clean

all: main

main: main.o funciones.o  
	$(CC) $(CFLAGS) $^ -o $@ 

funciones.o: funciones.cpp funciones.h
	$(CC) $(CFLAGS) -c $< -o $@


main.o: main.cpp funciones.h
	$(CC) $(CFLAGS) -c $< -o $@ 


.PHONY: clean

clean:
	rm -f *.o
	rm -f main