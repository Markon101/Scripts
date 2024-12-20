CXX = g++
CXXFLAGS = -std=c++11 -Wall -O3

all: solve

solver: solve.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

clean:
	rm -f solve

