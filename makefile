model: twoCptODE.o 
	gcc -g twoCptODE.o -o twoCptODE -lm

twoCptODE.o: twoCptODE.c 
	gcc -Wall -c -g twoCptODE.c

clean:
	rm -f twoCptODE *.o
