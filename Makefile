all:
	c++ -std=c++11 -march=native -O3 bmt.cc -funroll-loops -S -o bmt.s
	c++ -std=c++11 -march=native -O3 bmt.cc -funroll-loops -o bmt
	./bmt

clean:
	rm -f bmt bmt.s
